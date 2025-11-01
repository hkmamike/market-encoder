import pandas as pd
import awswrangler as wr
import boto3
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from transformers import AutoTokenizer, TFBertModel

# ---------------------------------
# Setup, verify access, and read training table
# ---------------------------------
print("--- Step 1: Libraries imported ---")

# --- IMPORTANT: Set these variables before running ---
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
# Define Siamese Model
# ---------------------------------
print(f"\n--- Step 6: Building Siamese Network ---")

def create_siamese_network(bert_model, max_length=MAX_LENGTH):
    """
    Creates the Siamese network architecture.
        """
    
    # --- Define Input Layers ---
    # Inputs for the first tower (Anchor)
    input_ids_1 = Input(shape=(max_length,), dtype='int32', name='input_ids_1')
    attention_mask_1 = Input(shape=(max_length,), dtype='int32', name='attention_mask_1')
    
    # Inputs for the second tower (Positive/Negative)
    input_ids_2 = Input(shape=(max_length,), dtype='int32', name='input_ids_2')
    attention_mask_2 = Input(shape=(max_length,), dtype='int32', name='attention_mask_2')

    # --- Define Shared Encoder ---
    # This is the core FinBERT model that will be shared
    # Set trainable=True to allow fine-tuning
    bert_encoder = TFBertModel.from_pretrained(MODEL_NAME, from_pt=True)
    bert_encoder.trainable = True
    
    # --- Process Inputs through Encoder ---
    # Tower 1
    output_1 = bert_encoder(input_ids=input_ids_1, attention_mask=attention_mask_1)
    # Use the [CLS] token's output (pooler_output) as the embedding
    embedding_1 = output_1.pooler_output 
    
    # Tower 2
    output_2 = bert_encoder(input_ids=input_ids_2, attention_mask=attention_mask_2)
    embedding_2 = output_2.pooler_output

    # --- Calculate Distance ---
    # Define a Lambda layer to compute the Euclidean distance
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
    
    # We return both the full model and the encoder
    encoder_model = Model(
        inputs=[input_ids_1, attention_mask_1], 
        outputs=embedding_1,
        name="encoder"
    )
    
    return model, encoder_model

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


# --- Instantiate the model ---
siamese_model, encoder_model = create_siamese_network(MODEL_NAME, MAX_LENGTH)
siamese_model.summary()


# ---------------------------------
# Compile and Train
# ---------------------------------
print(f"\n--- Step 7: Compiling and Training Model ---")

if df is not None:
    siamese_model.compile(
        loss=contrastive_loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5) 
        # Low learning rate is good for fine-tuning BERT
    )

    # Start training
    history = siamese_model.fit(
        X_train,
        y_train,
        epochs=3,  # Start with a few epochs
        batch_size=16, # Adjust based on GPU memory
        validation_split=0.2 # Use 20% of the data for validation
    )
    
    print("\nTraining complete.")

    # --- Save the encoder (Your "Stage 1" is complete) ---
    print("\n--- Step 8: Saving Encoder Model ---")
    encoder_model.save("finbert_encoder_model")
    print("Encoder model saved to 'finbert_encoder_model'.")
    print("You can now use this for the next stage of your project.")

else:
    print("\nSkipping training as DataFrame was not loaded.")