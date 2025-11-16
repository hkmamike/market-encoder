# ---------------------------------
# Same as model_v0_colab, trying to improve training speed by freezing earlier layer weights
# Status: Verified that it runs: https://screenshot.googleplex.com/72MM73yZDPE2JkY
# Training Time: 30s on Google Compute Backend Engine GPU
# Next: wait for real training data.
# ---------------------------------

# ---------------------------------
# Setup, verify access, and read training table
# ---------------------------------
!pip install awswrangler
!pip install tensorflow
# ---------------------------------

import pandas as pd
import awswrangler as wr
import boto3
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from transformers import AutoTokenizer, TFBertModel
import matplotlib.pyplot as plt
from getpass import getpass
import os

# ---------------------------------
# Setup, verify access, and read training table
# ---------------------------------
print("--- Step 1: Libraries imported ---")

# --- AWS Authentication for Colab ---
# Prompt for AWS credentials
aws_access_key_id = getpass('Enter AWS Access Key ID: ')
aws_secret_access_key = getpass('Enter AWS Secret Access Key: ')

# --- IMPORTANT: Set these variables before running ---
AWS_REGION = 'us-east-2'
S3_STAGING_DIR = 's3://cs230-market-data-2025/athena-query-results/'
ATHENA_DB = 'cs230_finance_data'
# Querying more data for a small training run
SQL_QUERY = "SELECT concatarticles1, concatarticles2, close_1_vs_2 FROM trainning_v0_example LIMIT 1000"

print(f"\n--- Step 3: Configuration set for {ATHENA_DB} ---")

# --- Model & Tokenizer Configuration ---
MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 256
# ----------------------------------------------------

# --- TPU Configuration ---
print("\n--- Step 2: Configuring TPU ---")
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    # If TPU is not available, check for GPU.
    print('⚠️ TPU not found. Checking for GPUs.')
    if tf.config.list_physical_devices('GPU'):
        # If GPUs are available, MirroredStrategy will use them all.
        # If only one GPU is available, it will use that one.
        strategy = tf.distribute.MirroredStrategy()
        print(f'✅ Running on {len(tf.config.list_physical_devices("GPU"))} GPU(s).')
    else:
        # If no GPU is found, fall back to CPU
        print('⚠️ No GPUs found. Running on CPU.')
        strategy = tf.distribute.get_strategy() # Default strategy for CPU

print(f"REPLICAS: {strategy.num_replicas_in_sync}")
# ----------------------------------------------------

print(f"--- Step 4: Querying Data ---")
print(f"Querying data from {ATHENA_DB}.trainning_v0_example...")

# Define df in a wider scope
df = None

try:
    # Create a boto3 session with the provided credentials
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=AWS_REGION,
    )

    # Run the query and load results into a Pandas DataFrame
    df = wr.athena.read_sql_query(
        sql=SQL_QUERY,
        database=ATHENA_DB,
        s3_output=S3_STAGING_DIR,
        boto3_session=session,
    )

    print("\nQuery successful! Data loaded into DataFrame.")

    # Display the first 5 rows
    print(df.head())

except Exception as e:
    print(f"\nAn error occurred:")
    print(e)


# ---------------------------------
# Load Tokenizer
# ---------------------------------
print(f"\n--- Step 5: Loading Tokenizer ({MODEL_NAME}) ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---------------------------------
# Preprocess & Tokenize Data
# ---------------------------------
print(f"\n--- Step 6: Tokenizing Data ---")
if df is not None:
    # Separate the text columns and the label
    text1_list = df['concatarticles1'].astype(str).tolist()
    text2_list = df['concatarticles2'].astype(str).tolist()
    # Ensure labels are float32 for the loss function
    labels = df['close_1_vs_2'].astype('float32').values

    # Tokenize both text lists
    print("Tokenizing concatarticles1...")
    encodings1 = tokenizer(
        text1_list,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )

    print("Tokenizing concatarticles2...")
    encodings2 = tokenizer(
        text2_list,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )

    # Create the input dictionary for the Keras model
    # This format matches the Input layers we will define
    X_train = {
        'input_ids_1': encodings1['input_ids'],
        'attention_mask_1': encodings1['attention_mask'],
        'input_ids_2': encodings2['input_ids'],
        'attention_mask_2': encodings2['attention_mask']
    }
    y_train = labels

    print(f"Data prepared: {len(y_train)} pairs.")

else:
    print("\nDataFrame is None. Halting script.")
    # In a real script, exit here
    # sys.exit()

# ---------------------------------
# Define Regression Model (using Subclassing API)
# ---------------------------------
print(f"\n--- Step 7: Building Regression Model ---")

class MarketDiffRegressor(Model):
    def __init__(self, bert_model_name, **kwargs):
        super(MarketDiffRegressor, self).__init__(**kwargs)
        # --- Define Shared Encoder ---
        self.bert_encoder = TFBertModel.from_pretrained(bert_model_name, from_pt=True)

        # --- Define Regression Head ---
        # We concatenate the two embeddings, so the input size is 2 * embedding_dim
        # FinBERT's embedding dim is 768. 768 * 2 = 1536
        self.regressor = Dense(1, activation='linear', name='regressor')
        
       # --- Freeze Layers ---
        # Crucial: Set the TFBertModel itself to trainable=True for its internal layers' trainable status to be respected by the parent Model
        self.bert_encoder.trainable = True

        # Now, explicitly freeze the layers we don't want to train
        num_bert_layers = len(self.bert_encoder.bert.encoder.layer)
        num_layers_to_unfreeze = 1 # Keep the last two layers trainable
        num_layers_to_freeze = num_bert_layers - num_layers_to_unfreeze

        for i, layer in enumerate(self.bert_encoder.bert.encoder.layer):
            if i < num_layers_to_freeze:
                layer.trainable = False
            else:
                layer.trainable = True # Explicitly ensure they are trainable

        print(f"\n--- Summarize Bert Encoder setting ---")

    def call(self, inputs):
        # Unpack the inputs
        input_ids_1 = inputs['input_ids_1']
        attention_mask_1 = inputs['attention_mask_1']
        input_ids_2 = inputs['input_ids_2']
        attention_mask_2 = inputs['attention_mask_2']
        
        # --- Process Inputs through Encoder ---
        # Tower 1
        output_1 = self.bert_encoder(input_ids=input_ids_1, attention_mask=attention_mask_1)
        embedding_1 = output_1.pooler_output

        # Tower 2
        output_2 = self.bert_encoder(input_ids=input_ids_2, attention_mask=attention_mask_2)
        embedding_2 = output_2.pooler_output

        # --- Concatenate embeddings and pass to regressor ---
        concatenated_embeddings = tf.concat([embedding_1, embedding_2], axis=1)
        prediction = self.regressor(concatenated_embeddings)
        
        return prediction

with strategy.scope():
    # --- Instantiate the model ---
    regression_model = MarketDiffRegressor(MODEL_NAME)
    
    # --- Build the model to create its weights ---
    if df is not None:
      # We need to build the model to see a summary and save it later
      # Pass a single sample through
      _ = regression_model(
          {
              'input_ids_1': X_train['input_ids_1'][:1],
              'attention_mask_1': X_train['attention_mask_1'][:1],
              'input_ids_2': X_train['input_ids_2'][:1],
              'attention_mask_2': X_train['attention_mask_2'][:1]
          }
      )

    # ---------------------------------
    # Compile and Train
    # ---------------------------------
    print(f"\n--- Step 8: Compiling and Training Model ---")

    regression_model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5)
    )

if df is not None:
    # Start training
    history = regression_model.fit(
        X_train,
        y_train,
        epochs=3,  # Start with a few epochs
        batch_size=16 * strategy.num_replicas_in_sync, # Adjust based on GPU memory
        validation_split=0.2 # Use 20% of the data for validation
    )

    print("\nTraining complete.")

    # ---------------------------------
    # Plot Training History
    # ---------------------------------
    print("\n--- Step 9: Plotting Training and Validation Loss ---")

    # Get loss and validation loss from history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot to a file
    plot_filename = 'regression_model_loss_plot.png'
    plt.savefig(plot_filename)
    print(f"Loss plot saved to {plot_filename}")
    # You can also use plt.show() if running in an interactive environment
    # plt.show()


    # ---------------------------------
    # Save Encoder Model
    # ---------------------------------
    # To save the encoder, you can create a separate model after training
    # print("\n--- Step 10: Saving Encoder Model ---")
    
    # # Create input layers for the encoder
    # input_ids = Input(shape=(MAX_LENGTH,), dtype='int32', name='input_ids')
    # attention_mask = Input(shape=(MAX_LENGTH,), dtype='int32', name='attention_mask')
    
    # # Create the encoder model using the trained weights
    # encoder_model = Model(
    #     inputs=[input_ids, attention_mask],
    #     outputs=regression_model.bert_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output,
    #     name="encoder"
    # )

    # encoder_model.save("finbert_encoder_model")
    # print("Encoder model saved to 'finbert_encoder_model'.")
    # print("You can now use this for the next stage of your project.")

else:
    print("\nSkipping training as DataFrame was not loaded.")

# Model Evaluation
# TODO: Add regression-specific evaluation (e.g., MAE, R^2 score)
print("\n--- Step 10: Evaluating on Validation Set ---")
print("Skipping detailed evaluation for this version. You can add regression metrics like MAE, MSE, or R-squared here.")


# Step 11: Save Models (Both Encoder and Full Siamese Weights)

print("\n--- Step 11: Saving Models ---")

# Create a directory for models if it doesn't exist
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# =========================================
# Save just the Encoder (Best for downstream use)
# =========================================
print("\n Saving standalone Encoder model...")
try:
    # 1. Define standard Keras inputs matching your tokenizer output exactly
    # Note: We use int32 match typical TF input requirements for indices
    enc_input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='input_ids')
    enc_attention_mask = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='attention_mask')

    # 2. Pass these inputs through the *trained* internal BERT encoder
    # We use named arguments to be safe with the Transformers library
    bert_output = regression_model.bert_encoder(
        input_ids=enc_input_ids,
        attention_mask=enc_attention_mask
    )

    # 3. Extract the pooler_output (standard for classification/similarity tasks)
    embeddings = bert_output.pooler_output

    # 4. Create the Functional model
    encoder_to_save = tf.keras.Model(
        inputs=[enc_input_ids, enc_attention_mask],
        outputs=embeddings,
        name='finbert_encoder'
    )

    # 6. Save in standard Keras format
    # Using .keras extension (recommended for TF 2.13+)
    encoder_save_path = os.path.join('saved_models', 'finbert_siamese_encoder.keras')
    encoder_to_save.save(encoder_save_path)
    print(f"SUCCESS: Encoder model saved to: {encoder_save_path}")
    print("Use this model to generate embeddings for new data.")

except Exception as e:
    print(f"FAILED to save encoder: {e}")

# =========================================
# Save full regression model weights (Best for resuming training)
# =========================================
print("Saving full regression model weights...")
try:
    # For subclassed models, saving weights is the most reliable method
    weights_save_path = os.path.join('saved_models', 'regression_model_weights.h5')
    regression_model.save_weights(weights_save_path)
    print(f"SUCCESS: Full regression model weights saved to: {weights_save_path}")
    print("To reload for more training: Instantiate a new MarketDiffRegressor and call .load_weights()")

except Exception as e:
    print(f"FAILED to save regression model weights: {e}")
