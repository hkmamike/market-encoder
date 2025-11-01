# ---------------------------------
# Same as model_v0, configured to work on colab
# ---------------------------------

# ---------------------------------
# Setup, verify access, and read training table
# ---------------------------------
!pip install awswrangler

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
    strategy = tf.distribute.get_strategy() # Default strategy that works on CPU and single GPU

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
        self.bert_encoder.trainable = True

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

    # ---------------------------------
    # Compile and Train
    # ---------------------------------
    print(f"\n--- Step 8: Compiling and Training Model ---")

    siamese_model.compile(
        loss=contrastive_loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5)
        # Low learning rate is good for fine-tuning BERT
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
    print("\n--- Step 10: Saving Encoder Model ---")
    
    # Create input layers for the encoder
    input_ids = Input(shape=(MAX_LENGTH,), dtype='int32', name='input_ids')
    attention_mask = Input(shape=(MAX_LENGTH,), dtype='int32', name='attention_mask')
    
    # Create the encoder model using the trained weights
    encoder_model = Model(
        inputs=[input_ids, attention_mask],
        outputs=siamese_model.bert_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output,
        name="encoder"
    )

    encoder_model.save("finbert_encoder_model")
    print("Encoder model saved to 'finbert_encoder_model'.")
    print("You can now use this for the next stage of your project.")

else:
    print("\nSkipping training as DataFrame was not loaded.")