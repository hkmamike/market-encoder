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
SQL_QUERY = "SELECT * FROM trainning_v0_example LIMIT 1000"

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
    labels = df['samebucket'].astype('float32').values

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
# Define Siamese Model (using Subclassing API)
# ---------------------------------
print(f"\n--- Step 7: Building Siamese Network ---")

class SiameseModel(Model):
    def __init__(self, bert_model_name, **kwargs):
        super(SiameseModel, self).__init__(**kwargs)
        # --- Define Shared Encoder ---
        self.bert_encoder = TFBertModel.from_pretrained(bert_model_name, from_pt=True)
        
        # --- Freeze Layers ---
        # Freeze the entire BERT model first
        self.bert_encoder.trainable = False
        
        # Unfreeze the last two layers
        num_layers_to_unfreeze = 2
        for layer in self.bert_encoder.bert.encoder.layer[-num_layers_to_unfreeze:]:
            layer.trainable = True

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

        # --- Calculate Distance ---
        sum_square = K.sum(K.square(embedding_1 - embedding_2), axis=1, keepdims=True)
        distance = K.sqrt(K.maximum(sum_square, K.epsilon()))
        
        return distance

# --- Define Contrastive Loss Function ---
def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss function.
    y_true: 1 if 'similar' (same bucket), 0 if 'dissimilar'
    y_pred: The Euclidean distance from the model
    """
    y_true = tf.cast(y_true, tf.float32)
    margin = 1.0

    # Loss for 'similar' pairs
    square_pred = K.square(y_pred)
    # Loss for 'dissimilar' pairs
    margin_square = K.square(K.maximum(0., margin - y_pred))

    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

with strategy.scope():
    # --- Instantiate the model ---
    siamese_model = SiameseModel(MODEL_NAME)
    
    # --- Build the model to create its weights ---
    if df is not None:
      _ = siamese_model(
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

    siamese_model.compile(
        loss=contrastive_loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5)
    )

if df is not None:
    # Start training
    history = siamese_model.fit(
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
    plot_filename = 'siamese_model_loss_plot.png'
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
    #     outputs=siamese_model.bert_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output,
    #     name="encoder"
    # )

    # encoder_model.save("finbert_encoder_model")
    # print("Encoder model saved to 'finbert_encoder_model'.")
    # print("You can now use this for the next stage of your project.")

else:
    print("\nSkipping training as DataFrame was not loaded.")

# Model Evaluation
# Todo: split the train and validation set before training. Also consider using test set.
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

print("\n--- Step 10: Evaluating on Validation Set ---")

if df is not None:
   # ---------------------------------
   # 1. Reconstruct the Validation Set
   # ---------------------------------
   # Keras 'validation_split' takes the LAST portion of the data.
   VAL_RATIO = 0.2
   split_idx = int(len(y_train) * (1 - VAL_RATIO))

   print(f"Recovering validation data from index {split_idx} to {len(y_train)}...")

   # Slice the dictionary inputs
   X_val = {
       'input_ids_1': X_train['input_ids_1'][split_idx:],
       'attention_mask_1': X_train['attention_mask_1'][split_idx:],
       'input_ids_2': X_train['input_ids_2'][split_idx:],
       'attention_mask_2': X_train['attention_mask_2'][split_idx:]
   }
   # Slice the labels
   y_val = y_train[split_idx:]

   # ---------------------------------
   # 2. Generate Distance Predictions
   # ---------------------------------
   print("Running predictions on validation set...")
   # The model outputs Euclidean distances
   val_distances = siamese_model.predict(X_val, batch_size=128, verbose=1)
   val_distances = val_distances.flatten()

   # ---------------------------------
   # 3. Determine Optimal Threshold
   # ---------------------------------
   # A Siamese network outputs distances, not classes. We need a threshold.
   # If distance < threshold -> Predict 'Similar' (1)
   # If distance >= threshold -> Predict 'Dissimilar' (0)

   print("Finding optimal distance threshold based on F1 score...")
   thresholds = np.arange(0.1, 3.0, 0.05)
   best_f1 = -1
   best_threshold = 0.5

   f1_scores = []
   for t in thresholds:
       # Predict 1 if distance is small (less than t)
       preds_t = (val_distances < t).astype(np.float32)
       score = f1_score(y_val, preds_t, zero_division=0)
       f1_scores.append(score)
       if score > best_f1:
           best_f1 = score
           best_threshold = t

   print(f"\nBest Threshold: {best_threshold:.2f}")
   print(f"Best Validation F1: {best_f1:.4f}")

   # Optional: Plot F1 vs Threshold to see stability
   plt.figure(figsize=(8, 4))
   plt.plot(thresholds, f1_scores, 'b-')
   plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold ({best_threshold:.2f})')
   plt.title("F1 Score vs Distance Threshold")
   plt.xlabel("Distance Threshold")
   plt.ylabel("F1 Score")
   plt.legend()
   plt.savefig('threshold_f1_plot.png')
   print("Threshold plot saved to 'threshold_f1_plot.png'")

   # ---------------------------------
   # 4. Final Evaluation Metrics
   # ---------------------------------
   # Generate final class predictions using the best threshold
   val_predictions = (val_distances < best_threshold).astype(np.float32)

   print("\n--- Final Validation Performance Report ---")
   print(f"Threshold used: {best_threshold:.2f}")
   print("\nConfusion Matrix:")
   # Format: [[TN, FP], [FN, TP]]
   print(confusion_matrix(y_val, val_predictions))

   print("\nClassification Report:")
   print(classification_report(y_val, val_predictions, target_names=['Dissimilar (0)', 'Similar (1)']))

else:
   print("Cannot evaluate: DataFrame was not loaded.")


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
    bert_output = siamese_model.bert_encoder(
        input_ids=enc_input_ids,
        attention_mask=enc_attention_mask
    )

    # 3. Extract the pooler_output (standard for classification/similarity tasks)
    embeddings = bert_output.pooler_output

    # 4. IMPORTANT: Replicate the L2 normalization from your SiameseModel.call()
    # If you don't do this, the distances you calculate later will be wrong.
    normalized_embeddings = tf.linalg.normalize(embeddings, axis=1)[0]

    # 5. Create the Functional model
    encoder_to_save = tf.keras.Model(
        inputs=[enc_input_ids, enc_attention_mask],
        outputs=normalized_embeddings,
        name='finbert_encoder_normalized'
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
# Save full Siamese model weights (Best for resuming training)
# =========================================
print("Saving full Siamese model weights...")
try:
    # For subclassed models, saving weights is the most reliable method
    weights_save_path = os.path.join('saved_models', 'siamese_complete_weights.h5')
    siamese_model.save_weights(weights_save_path)
    print(f"SUCCESS: Full Siamese weights saved to: {weights_save_path}")
    print("To reload for more training: Instantiate a new SiameseModel and call .load_weights()")

except Exception as e:
    print(f"FAILED to save Siamese weights: {e}")
