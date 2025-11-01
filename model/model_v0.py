# Add these lines at the very top
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING logs

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

# ---------------------------------
# Setup, verify access, and read training table
# ---------------------------------
print("--- Step 1: Libraries imported ---")

# ---------------------------------
# Initialize Distribution Strategy (TPU / GPU / CPU)
# ---------------------------------
print("\n--- Step 1.5: Initializing Distribution Strategy ---")
try:
    # Attempt to connect to a TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() 
    strategy = tf.distribute.TPUStrategy(tpu)
    print("✅ TPU found and connected.")
    print('Running on TPU:', tpu.master())
except ValueError:
    # If TPU not found, fall back to the default strategy
    # This will use MirroredStrategy for multiple GPUs, or
    # OneDeviceStrategy for a single GPU or CPU.
    strategy = tf.distribute.get_strategy()
    print("⚠️ No TPU found. Using default strategy (GPU/CPU).")

print(f"Strategy: {strategy.__class__.__name__}")
print(f"Number of replicas: {strategy.num_replicas_in_sync}")


# ---------------------------------
# --- IMPORTANT: Set these variables before running ---
# ---------------------------------
AWS_REGION = 'us-east-2'
S3_STAGING_DIR = 's3://cs230-market-data-2025/athena-query-results/'
ATHENA_DB = 'cs230_finance_data'
# Querying more data for a small training run
SQL_QUERY = "SELECT * FROM trainning_v0_example LIMIT 1000" 
AWS_PROFILE_NAME = "team-s3-uploader"

# --- Model & Tokenizer Configuration ---
MODEL_NAME = "ProsusAI/tiny-finbert"
MAX_LENGTH = 256 
# ----------------------------------------------------

print(f"--- Step 2: Configuration set for {ATHENA_DB} ---")
print(f"--- Step 3: Querying Data ---")
print(f"Querying data from {ATHENA_DB}.trainning_v0_example...")

# Define df in a wider scope
df = None 

try:
    # Create a boto3 session with the specified region
    session = boto3.Session(profile_name=AWS_PROFILE_NAME, region_name=AWS_REGION)

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
print(f"\n--- Step 4: Loading Tokenizer ({MODEL_NAME}) ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---------------------------------
# Preprocess & Tokenize Data
# ---------------------------------
print(f"\n--- Step 5: Tokenizing Data ---")
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
# Step 6: Define Model & Loss Functions
# ---------------------------------
print(f"\n--- Step 6: Defining Model & Loss Functions ---")

def create_siamese_network(bert_model_name, max_length=MAX_LENGTH, num_layers_to_unfreeze=1):
    """
    Creates the Siamese network architecture.
    """
    
    # --- Define Input Layers ---
    input_ids_1 = Input(shape=(max_length,), dtype='int32', name='input_ids_1')
    attention_mask_1 = Input(shape=(max_length,), dtype='int32', name='attention_mask_1')
    input_ids_2 = Input(shape=(max_length,), dtype='int32', name='input_ids_2')
    attention_mask_2 = Input(shape=(max_length,), dtype='int32', name='attention_mask_2')

    # --- Define Shared Encoder ---
    bert_encoder = TFBertModel.from_pretrained(bert_model_name, from_pt=True)
    
    # --- Freeze layers ---
    bert_encoder.trainable = False
    if num_layers_to_unfreeze > 0:
        for layer in bert_encoder.encoder.layer[-num_layers_to_unfreeze:]:
            layer.trainable = True
    if bert_encoder.pooler is not None:
        bert_encoder.pooler.trainable = True
    
    # --- Process Inputs through Encoder ---
    output_1 = bert_encoder(input_ids=input_ids_1, attention_mask=attention_mask_1)
    embedding_1 = output_1.pooler_output 
    
    output_2 = bert_encoder(input_ids=input_ids_2, attention_mask=attention_mask_2)
    embedding_2 = output_2.pooler_output

    # --- Calculate Distance ---
    def euclidean_distance(vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    distance = Lambda(euclidean_distance, name='euclidean_distance')([embedding_1, embedding_2])

    # --- Create and return the Keras Model ---
    model = Model(
        inputs=[input_ids_1, attention_mask_1, input_ids_2, attention_mask_2],
        outputs=distance,
        name="siamese_network"
    )
    
    encoder_model = Model(
        inputs=[input_ids_1, attention_mask_1], 
        outputs=embedding_1,
        name="encoder"
    )
    
    return model, encoder_model

# --- Define Contrastive Loss Function ---
def contrastive_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    margin = 1.0
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(0., margin - y_pred))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


# ---------------------------------
# NEW: Step 7: Build and Compile Model under Strategy Scope
# ---------------------------------
print(f"\n--- Step 7: Building & Compiling Model under Strategy Scope ---")

# We must create the model and optimizer inside the 'strategy.scope()'
# so that TensorFlow knows to create the variables on the TPU/replicas.
with strategy.scope():
    # Instantiate the model
    # We will unfreeze the last 2 layers (you can tune this)
    siamese_model, encoder_model = create_siamese_network(MODEL_NAME, MAX_LENGTH, num_layers_to_unfreeze=2)
    
    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    
    # Compile the model
    siamese_model.compile(
        loss=contrastive_loss,
        optimizer=optimizer
    )

# Print the summary outside the scope
siamese_model.summary()


# ---------------------------------
# Step 8: Verify Trainable Layers
# ---------------------------------
print(f"\n--- Step 8: Checking Trainable Layers in BERT Encoder ---")
try:
    bert_layer = siamese_model.get_layer('tf_bert_model') 
    print(f"Total layers in BERT: {len(bert_layer.encoder.layer)}")
    print(f"Pooler Trainable: {bert_layer.pooler.trainable}")
    for i, layer in enumerate(bert_layer.encoder.layer):
        print(f"Layer {i} trainable: {layer.trainable}")
except Exception as e:
    print(f"Could not print layer status: {e}")


# ---------------------------------
# Step 9: Train Model
# ---------------------------------
print(f"\n--- Step 9: Training Model ---")

if df is not None:
    # Start training
    # .fit() is automatically strategy-aware
    history = siamese_model.fit(
        X_train,
        y_train,
        epochs=3,  
        batch_size=16 * strategy.num_replicas_in_sync, # Scale batch size
        validation_split=0.2
    )
    
    print("\nTraining complete.")

    # ---------------------------------
    # Step 10: Plot Training History
    # ---------------------------------
    print("\n--- Step 10: Plotting Training and Validation Loss ---")
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plot_filename = 'siamese_model_loss_plot.png'
    plt.savefig(plot_filename)
    print(f"Loss plot saved to {plot_filename}")

    # ---------------------------------
    # Step 11: Save Encoder Model
    # ---------------------------------
    print("\n--- Step 11: Saving Encoder Model ---")
    # Saving should be done outside the scope
    encoder_model.save("finbert_encoder_model")
    print("Encoder model saved to 'finbert_encoder_model'.")

else:
    print("\nSkipping training as DataFrame was not loaded.")