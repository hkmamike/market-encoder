# ---------------------------------
# Baseline model
# Input: 10 concat news titles
# Output: closing price of vixy of the day
# Training Time: 30s on Google Compute Backend Engine GPU
# Next: Train on full dataset and evaluate regression metrics.
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
SQL_QUERY = "SELECT concatarticles1, close1 FROM paired_vixy_w_titles_dedup_more_labels LIMIT 1000"

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
print(f"Querying data from {ATHENA_DB}.paired_vixy_w_titles_dedup_more_labels...")

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
    text_list = df['concatarticles1'].astype(str).tolist()
    # Ensure labels are float32 for the loss function
    labels = df['close1'].astype('float32').values

    # Tokenize the text list
    print("Tokenizing concatarticles1...")
    encodings = tokenizer(
        text_list,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )

    # Create the input dictionary for the Keras model
    X_train = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask']
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

class MarketRegressorModel(Model):
    def __init__(self, bert_model_name, **kwargs):
        super(MarketRegressorModel, self).__init__(**kwargs)
        # --- Define Shared Encoder ---
        self.bert_encoder = TFBertModel.from_pretrained(bert_model_name, from_pt=True)
        
        # --- Define Regression Head ---
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

        # Usually, embeddings are frozen in fine-tuning for BERT models
        self.bert_encoder.bert.embeddings.trainable = False
        print(f"\n--- Summarize Bert Encoder setting ---")

    def call(self, inputs):
        # Unpack the inputs
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # --- Process Inputs through Encoder ---
        output = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
        embedding = output.pooler_output
        
        # --- Pass embedding to the regression head ---
        prediction = self.regressor(embedding)
        
        return prediction

with strategy.scope():
    # --- Instantiate the model ---
    regressor_model = MarketRegressorModel(MODEL_NAME)
    
    # --- Build the model to create its weights ---
    if df is not None:
      _ = regressor_model(
          {
              'input_ids': X_train['input_ids'][:1],
              'attention_mask': X_train['attention_mask'][:1]
          }
      )

    # ---------------------------------
    # Compile and Train
    # ---------------------------------
    print(f"\n--- Step 8: Compiling and Training Model ---")

    regressor_model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5)
    )

if df is not None:
    # Start training
    history = regressor_model.fit(
        X_train,
        y_train,
        epochs=3,  # Start with a few epochs
        batch_size=64 * strategy.num_replicas_in_sync, # Adjust based on GPU memory
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

else:
    print("\nSkipping training as DataFrame was not loaded.")