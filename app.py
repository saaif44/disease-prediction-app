# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import os
import re
from fuzzywuzzy import process, fuzz # For NLP-like symptom extraction

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Configuration & Paths ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'disease_prediction_model.pkl')
SYMPTOM_COLUMNS_PATH = os.path.join(MODEL_DIR, 'symptom_columns.pkl')

DOCTORS_CSV_PATH = os.path.join(BASE_DIR, 'doctors_bd_detailed.csv')
DISEASE_DESC_CSV_PATH = os.path.join(BASE_DIR, 'symptom_Description.csv')
DISEASE_PRECAUTION_CSV_PATH = os.path.join(BASE_DIR, 'symptom_precaution.csv')

# --- Global Variables for Loaded Data ---
model = None
MODEL_SYMPTOM_KEYS = []
doctors_df = pd.DataFrame()
disease_desc_df = pd.DataFrame()
disease_precaution_df = pd.DataFrame()

# --- SYMPTOM MAP (CRITICAL: Populate this thoroughly!) ---
SYMPTOM_MAP = {
    # ... (your existing SYMPTOM_MAP - keep it as is) ...
    "itching": {"model_key": "itching", "ask_phrase": "Are you experiencing any itching?"},
    "skin rash": {"model_key": "skin_rash", "ask_phrase": "Do you have a skin rash or any rashes on your skin?"},
    "nodal skin eruptions": {"model_key": "nodal_skin_eruptions", "ask_phrase": "Have you noticed any nodal skin eruptions, like bumps under the skin?"},
    "continuous sneezing": {"model_key": "continuous_sneezing", "ask_phrase": "Are you sneezing continuously or very frequently?"},
    "shivering": {"model_key": "shivering", "ask_phrase": "Are you shivering, perhaps feeling cold even when it's not?"},
    "chills": {"model_key": "chills", "ask_phrase": "Do you have chills, possibly with a fever?"},
    "joint pain": {"model_key": "joint_pain", "ask_phrase": "Are you experiencing pain in your joints?"},
    "stomach pain": {"model_key": "stomach_pain", "ask_phrase": "Do you have stomach pain or an ache in your abdomen?"},
    "acidity": {"model_key": "acidity", "ask_phrase": "Are you suffering from acidity or heartburn?"},
    "ulcers on tongue": {"model_key": "ulcers_on_tongue", "ask_phrase": "Do you have any ulcers or sores on your tongue?"},
    "muscle wasting": {"model_key": "muscle_wasting", "ask_phrase": "Have you noticed any muscle wasting or a decrease in muscle mass?"},
    "vomiting": {"model_key": "vomiting", "ask_phrase": "Have you been vomiting or throwing up?"},
    "burning micturition": {"model_key": "burning_micturition", "ask_phrase": "Do you feel a burning sensation when you urinate?"},
    # "spotting urination": {"model_key": "spotting_ urination", "ask_phrase": "Have you noticed any spotting or unusual discharge during urination?"}, # Assuming space converted to underscore for model_key
    "fatigue": {"model_key": "fatigue", "ask_phrase": "Are you feeling fatigued, very tired, or lacking energy?"},
    "weight gain": {"model_key": "weight_gain", "ask_phrase": "Have you experienced unexplained weight gain recently?"},
    "anxiety": {"model_key": "anxiety", "ask_phrase": "Are you feeling anxious, worried, or uneasy?"},
    "cold hands and feet": {"model_key": "cold_hands_and_feets", "ask_phrase": "Do your hands and feet often feel cold?"}, # Original key 'feets'
    "mood swings": {"model_key": "mood_swings", "ask_phrase": "Are you experiencing frequent mood swings or changes in your emotional state?"},
    "weight loss": {"model_key": "weight_loss", "ask_phrase": "Have you had unexplained weight loss recently?"},
    "restlessness": {"model_key": "restlessness", "ask_phrase": "Are you feeling restless or unable to relax?"},
    "lethargy": {"model_key": "lethargy", "ask_phrase": "Are you experiencing lethargy or a lack of energy and enthusiasm?"},
    "patches in throat": {"model_key": "patches_in_throat", "ask_phrase": "Have you noticed any patches or unusual spots in your throat?"},
    "irregular sugar level": {"model_key": "irregular_sugar_level", "ask_phrase": "Have you had irregular blood sugar levels?"},
    "cough": {"model_key": "cough", "ask_phrase": "Do you have a cough?"},
    "high fever": {"model_key": "high_fever", "ask_phrase": "Do you have a high fever?"},
    "sunken eyes": {"model_key": "sunken_eyes", "ask_phrase": "Do your eyes appear sunken or hollow?"},
    "breathlessness": {"model_key": "breathlessness", "ask_phrase": "Are you experiencing breathlessness or shortness of breath?"},
    "sweating": {"model_key": "sweating", "ask_phrase": "Are you sweating more than usual, or having night sweats?"},
    "dehydration": {"model_key": "dehydration", "ask_phrase": "Do you feel dehydrated, thirsty, or have a dry mouth?"},
    "indigestion": {"model_key": "indigestion", "ask_phrase": "Are you suffering from indigestion or an upset stomach after eating?"},
    "headache": {"model_key": "headache", "ask_phrase": "Do you have a headache?"},
    "yellowish skin": {"model_key": "yellowish_skin", "ask_phrase": "Has your skin taken on a yellowish tint?"},
    "dark urine": {"model_key": "dark_urine", "ask_phrase": "Is your urine darker than usual?"},
    "nausea": {"model_key": "nausea", "ask_phrase": "Are you feeling nauseous or like you might vomit?"},
    "loss of appetite": {"model_key": "loss_of_appetite", "ask_phrase": "Have you experienced a loss of appetite?"},
    "pain behind the eyes": {"model_key": "pain_behind_the_eyes", "ask_phrase": "Do you have pain behind your eyes?"},
    "back pain": {"model_key": "back_pain", "ask_phrase": "Are you experiencing back pain?"},
    "constipation": {"model_key": "constipation", "ask_phrase": "Are you suffering from constipation?"},
    "abdominal pain": {"model_key": "abdominal_pain", "ask_phrase": "Do you have pain in your abdomen (belly area)?"},
    "diarrhoea": {"model_key": "diarrhoea", "ask_phrase": "Are you experiencing diarrhoea or loose stools?"},
    "mild fever": {"model_key": "mild_fever", "ask_phrase": "Do you have a mild fever?"},
    "yellow urine": {"model_key": "yellow_urine", "ask_phrase": "Is your urine distinctly yellow?"}, # Note: dark_urine already exists, this might be redundant or different context
    "yellowing of eyes": {"model_key": "yellowing_of_eyes", "ask_phrase": "Have the whites of your eyes turned yellow?"},
    "acute liver failure": {"model_key": "acute_liver_failure", "ask_phrase": "Are there signs or a diagnosis of acute liver failure?"}, # This is serious, bot should emphasize doctor visit
    "fluid overload": {"model_key": "fluid_overload", "ask_phrase": "Are you experiencing symptoms of fluid overload, like swelling?"}, # If another "fluid_overload" exists, ensure distinct model_keys
    "swelling of stomach": {"model_key": "swelling_of_stomach", "ask_phrase": "Is your stomach swollen or distended?"},
    "swelled lymph nodes": {"model_key": "swelled_lymph_nodes", "ask_phrase": "Do you have any swelled lymph nodes, for example, in your neck, armpits, or groin?"},
    "malaise": {"model_key": "malaise", "ask_phrase": "Are you feeling a general sense of malaise, discomfort, or illness?"},
    "blurred and distorted vision": {"model_key": "blurred_and_distorted_vision", "ask_phrase": "Is your vision blurred or distorted?"},
    "phlegm": {"model_key": "phlegm", "ask_phrase": "Are you coughing up phlegm or mucus?"},
    "throat irritation": {"model_key": "throat_irritation", "ask_phrase": "Do you have throat irritation or a scratchy throat?"},
    "redness of eyes": {"model_key": "redness_of_eyes", "ask_phrase": "Are your eyes red or bloodshot?"},
    "sinus pressure": {"model_key": "sinus_pressure", "ask_phrase": "Do you feel pressure in your sinuses?"},
    "runny nose": {"model_key": "runny_nose", "ask_phrase": "Do you have a runny nose?"},
    "congestion": {"model_key": "congestion", "ask_phrase": "Are you experiencing nasal congestion or a stuffy nose?"},
    "chest pain": {"model_key": "chest_pain", "ask_phrase": "Are you experiencing any chest pain?"}, # Critical, emphasize doctor
    "weakness in limbs": {"model_key": "weakness_in_limbs", "ask_phrase": "Do you have weakness in your arms or legs?"},
    "fast heart rate": {"model_key": "fast_heart_rate", "ask_phrase": "Is your heart beating faster than usual, or do you have palpitations?"},
    "pain during bowel movements": {"model_key": "pain_during_bowel_movements", "ask_phrase": "Do you experience pain during bowel movements?"},
    "pain in anal region": {"model_key": "pain_in_anal_region", "ask_phrase": "Do you have pain in your anal region?"},
    "bloody stool": {"model_key": "bloody_stool", "ask_phrase": "Have you noticed any blood in your stool?"},
    "irritation in anus": {"model_key": "irritation_in_anus", "ask_phrase": "Do you have irritation in your anus?"},
    "neck pain": {"model_key": "neck_pain", "ask_phrase": "Are you experiencing neck pain?"},
    "dizziness": {"model_key": "dizziness", "ask_phrase": "Are you feeling dizzy or lightheaded?"},
    "cramps": {"model_key": "cramps", "ask_phrase": "Are you experiencing cramps (e.g., muscle or abdominal)?"},
    "bruising": {"model_key": "bruising", "ask_phrase": "Are you bruising more easily than usual?"},
    "obesity": {"model_key": "obesity", "ask_phrase": "Are you concerned about obesity or significant overweight?"}, # This is a condition, not a typical acute symptom
    "swollen legs": {"model_key": "swollen_legs", "ask_phrase": "Are your legs swollen?"},
    "swollen blood vessels": {"model_key": "swollen_blood_vessels", "ask_phrase": "Have you noticed any swollen blood vessels?"},
    "puffy face and eyes": {"model_key": "puffy_face_and_eyes", "ask_phrase": "Is your face or around your eyes puffy?"},
    "enlarged thyroid": {"model_key": "enlarged_thyroid", "ask_phrase": "Do you have an enlarged thyroid or a noticeable swelling in the front of your neck?"},
    "brittle nails": {"model_key": "brittle_nails", "ask_phrase": "Are your nails brittle or breaking easily?"},
    "swollen extremeties": {"model_key": "swollen_extremeties", "ask_phrase": "Are your extremeties (hands, feet, arms, legs) swollen?"}, # Note: extremities
    "excessive hunger": {"model_key": "excessive_hunger", "ask_phrase": "Are you experiencing excessive hunger?"},
    "extra marital contacts": {"model_key": "extra_marital_contacts", "ask_phrase": "Have you had extra-marital contacts? (This information is confidential and helps assess certain risks.)"}, # Sensitive, handle with care
    "drying and tingling lips": {"model_key": "drying_and_tingling_lips", "ask_phrase": "Are your lips dry or do they have a tingling sensation?"},
    "slurred speech": {"model_key": "slurred_speech", "ask_phrase": "Is your speech slurred or difficult to understand?"}, # Critical
    "knee pain": {"model_key": "knee_pain", "ask_phrase": "Are you experiencing knee pain?"},
    "hip joint pain": {"model_key": "hip_joint_pain", "ask_phrase": "Do you have pain in your hip joint?"},
    "muscle weakness": {"model_key": "muscle_weakness", "ask_phrase": "Are you experiencing muscle weakness?"},
    "stiff neck": {"model_key": "stiff_neck", "ask_phrase": "Do you have a stiff neck?"},
    "swelling joints": {"model_key": "swelling_joints", "ask_phrase": "Are any of your joints swollen?"},
    "movement stiffness": {"model_key": "movement_stiffness", "ask_phrase": "Do you feel stiffness when trying to move?"},
    "spinning movements": {"model_key": "spinning_movements", "ask_phrase": "Are you experiencing spinning sensations or vertigo?"},
    "loss of balance": {"model_key": "loss_of_balance", "ask_phrase": "Have you had any loss of balance or unsteadiness?"},
    "unsteadiness": {"model_key": "unsteadiness", "ask_phrase": "Do you feel unsteady on your feet?"},
    "weakness of one body side": {"model_key": "weakness_of_one_body_side", "ask_phrase": "Do you have weakness on one side of your body?"}, # Critical
    "loss of smell": {"model_key": "loss_of_smell", "ask_phrase": "Have you experienced a loss of smell?"},
    "bladder discomfort": {"model_key": "bladder_discomfort", "ask_phrase": "Are you feeling any discomfort in your bladder area?"},
    # "foul smell of urine": {"model_key": "foul_smell_of urine", "ask_phrase": "Does your urine have a foul or unusual smell?"}, # Assuming space to underscore
    "continuous feel of urine": {"model_key": "continuous_feel_of_urine", "ask_phrase": "Do you have a continuous feeling of needing to urinate?"},
    "passage of gases": {"model_key": "passage_of_gases", "ask_phrase": "Are you passing more gas than usual?"},
    "internal itching": {"model_key": "internal_itching", "ask_phrase": "Are you experiencing internal itching (e.g., vaginal or anal)?"},
    "toxic look (typhos)": {"model_key": "toxic_look_(typhos)", "ask_phrase": "Do you appear very ill, perhaps with a 'toxic look' or typhos-like state?"}, # TYPHOS is a severe state
    "depression": {"model_key": "depression", "ask_phrase": "Are you feeling depressed, sad, or hopeless?"},
    "irritability": {"model_key": "irritability", "ask_phrase": "Are you feeling more irritable than usual?"},
    "muscle pain": {"model_key": "muscle_pain", "ask_phrase": "Are you experiencing muscle pain or aches?"},
    "altered sensorium": {"model_key": "altered_sensorium", "ask_phrase": "Have you experienced any altered sensorium, confusion, or changes in consciousness?"}, # Critical
    "red spots over body": {"model_key": "red_spots_over_body", "ask_phrase": "Have you noticed red spots appearing over your body?"},
    "belly pain": {"model_key": "belly_pain", "ask_phrase": "Do you have pain in your belly area?"}, # Note: 'stomach_pain' and 'abdominal_pain' exist. Ensure model keys are distinct if these are truly different symptoms in model.
    "abnormal menstruation": {"model_key": "abnormal_menstruation", "ask_phrase": "Are you experiencing abnormal menstruation (e.g., irregular, heavy, or missed periods)?"},
    # "dischromic patches": {"model_key": "dischromic _patches", "ask_phrase": "Do you have dischromic patches (discolored skin patches)?"}, # Assuming dischromic__patches -> dischromic_patches
    "watering from eyes": {"model_key": "watering_from_eyes", "ask_phrase": "Are your eyes watering excessively?"},
    "increased appetite": {"model_key": "increased_appetite", "ask_phrase": "Has your appetite increased significantly?"},
    "polyuria": {"model_key": "polyuria", "ask_phrase": "Are you experiencing polyuria (urinating large volumes frequently)?"},
    "family history": {"model_key": "family_history", "ask_phrase": "Is there a family history of similar conditions or specific diseases?"}, # This is a risk factor, not a direct symptom usually
    "mucoid sputum": {"model_key": "mucoid_sputum", "ask_phrase": "Are you coughing up mucoid (clear or white) sputum?"},
    "rusty sputum": {"model_key": "rusty_sputum", "ask_phrase": "Are you coughing up rusty-colored sputum?"},
    "lack of concentration": {"model_key": "lack_of_concentration", "ask_phrase": "Are you having difficulty concentrating?"},
    "visual disturbances": {"model_key": "visual_disturbances", "ask_phrase": "Are you experiencing any visual disturbances other than blurred vision?"},
    "receiving blood transfusion": {"model_key": "receiving_blood_transfusion", "ask_phrase": "Have you recently received a blood transfusion?"}, # Risk factor
    "receiving unsterile injections": {"model_key": "receiving_unsterile_injections", "ask_phrase": "Have you recently received any unsterile injections?"}, # Risk factor
    "coma": {"model_key": "coma", "ask_phrase": "Has there been any instance of coma or unresponsiveness?"}, # Critical
    "stomach bleeding": {"model_key": "stomach_bleeding", "ask_phrase": "Are there any signs of stomach bleeding (e.g., vomiting blood, black tarry stools)?"}, # Critical
    "distention of abdomen": {"model_key": "distention_of_abdomen", "ask_phrase": "Is your abdomen distended or significantly bloated?"},
    "history of alcohol consumption": {"model_key": "history_of_alcohol_consumption", "ask_phrase": "Do you have a history of significant alcohol consumption?"}, # Risk factor
    # "fluid_overload" is listed twice in your input. Assuming it maps to the same model_key. If they are different symptoms in the model, they need different keys.
    # Assuming the second one is the same or a typo.
    "blood in sputum": {"model_key": "blood_in_sputum", "ask_phrase": "Are you coughing up blood in your sputum?"},
    "prominent veins on calf": {"model_key": "prominent_veins_on_calf", "ask_phrase": "Are the veins on your calf prominent or bulging?"},
    "palpitations": {"model_key": "palpitations", "ask_phrase": "Are you experiencing palpitations or a fluttering sensation in your chest?"}, # Similar to fast_heart_rate, check model diff.
    "painful walking": {"model_key": "painful_walking", "ask_phrase": "Is it painful for you to walk?"},
    "pus filled pimples": {"model_key": "pus_filled_pimples", "ask_phrase": "Do you have pus-filled pimples?"},
    "blackheads": {"model_key": "blackheads", "ask_phrase": "Are you experiencing blackheads?"},
    "scurring": {"model_key": "scurring", "ask_phrase": "Do you have scurring (scarring or scabbing) on your skin?"}, # Spelling "scurring" as in dataset
    "skin peeling": {"model_key": "skin_peeling", "ask_phrase": "Is your skin peeling?"},
    "silver like dusting": {"model_key": "silver_like_dusting", "ask_phrase": "Do you have a silver-like dusting or scales on your skin?"},
    "small dents in nails": {"model_key": "small_dents_in_nails", "ask_phrase": "Are there small dents or pits in your nails?"},
    "inflammatory nails": {"model_key": "inflammatory_nails", "ask_phrase": "Are your nails inflamed, red, or swollen around the edges?"},
    "blister": {"model_key": "blister", "ask_phrase": "Have you developed any blisters on your skin?"},
    "red sore around nose": {"model_key": "red_sore_around_nose", "ask_phrase": "Do you have a red sore or irritation around your nose?"},
    "yellow crust ooze": {"model_key": "yellow_crust_ooze", "ask_phrase": "Is there any yellow crust or ooze from skin lesions?"},

    # --- Synonyms (Add more as you think of them) ---
    "sore throat": {"model_key": "throat_irritation", "ask_phrase": "Do you have a sore throat?"}, # Or 'patches_in_throat' depending on context
    "tiredness": {"model_key": "fatigue", "ask_phrase": "Are you feeling extremely tired?"},
    "throwing up": {"model_key": "vomiting", "ask_phrase": "Have you been throwing up?"},
    "painful urination": {"model_key": "burning_micturition", "ask_phrase": "Is it painful when you urinate?"},
    "loose stools": {"model_key": "diarrhoea", "ask_phrase": "Are you having loose stools or diarrhoea?"},
    "shortness of breath": {"model_key": "breathlessness", "ask_phrase": "Are you experiencing shortness of breath?"},
    "upset stomach": {"model_key": "indigestion", "ask_phrase": "Do you have an upset stomach?"}, # Or nausea/vomiting depending on detail
    "feeling sick": {"model_key": "nausea", "ask_phrase": "Are you feeling sick to your stomach?"},
    "stuffy nose": {"model_key": "congestion", "ask_phrase": "Do you have a stuffy nose?"},
    "vertigo": {"model_key": "spinning_movements", "ask_phrase": "Are you experiencing vertigo or a spinning sensation?"},
    "passing gas": {"model_key": "passage_of_gases", "ask_phrase": "Are you passing more gas than usual?"},
    "feeling down": {"model_key": "depression", "ask_phrase": "Are you feeling down or depressed?"},
    "peeing a lot": {"model_key": "polyuria", "ask_phrase": "Are you urinating much more frequently or in larger amounts?"},
}

MODEL_KEY_TO_ASK_PHRASE = {}
NATURAL_SYMPTOM_PHRASES_FOR_FUZZY = []


def initialize_app_data():
    global model, MODEL_SYMPTOM_KEYS, doctors_df, disease_desc_df, disease_precaution_df
    global MODEL_KEY_TO_ASK_PHRASE, NATURAL_SYMPTOM_PHRASES_FOR_FUZZY

    app.logger.info("Initializing application data...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SYMPTOM_COLUMNS_PATH, 'rb') as f:
            MODEL_SYMPTOM_KEYS = pickle.load(f)
        MODEL_SYMPTOM_KEYS = [key.strip().lower().replace(' ', '_') for key in MODEL_SYMPTOM_KEYS]
        app.logger.info(f"Model and {len(MODEL_SYMPTOM_KEYS)} symptom keys loaded.")

        app.logger.info(f"Verifying SYMPTOM_MAP against {len(MODEL_SYMPTOM_KEYS)} model keys...")
        validated_symptom_map_temp = {}
        valid_symptom_map_natural_phrases = [] # Not used directly later, but good for debugging

        for natural_phrase, details in SYMPTOM_MAP.items():
            model_key_in_map = details.get("model_key")
            if not isinstance(model_key_in_map, str):
                 app.logger.warning(f"SymptomMap Integrity Issue: Entry for '{natural_phrase}' has no 'model_key' or it's not a string. Skipping.")
                 continue
            normalized_model_key_in_map = model_key_in_map.strip().lower().replace(' ', '_')
            if normalized_model_key_in_map not in MODEL_SYMPTOM_KEYS:
                app.logger.warning(f"SymptomMap Error: model_key '{normalized_model_key_in_map}' (from phrase '{natural_phrase}') not found in loaded MODEL_SYMPTOM_KEYS. Skipping this entry.")
            else:
                details["model_key"] = normalized_model_key_in_map
                validated_symptom_map_temp[natural_phrase] = details
                valid_symptom_map_natural_phrases.append(natural_phrase)
        
        SYMPTOM_MAP.clear()
        SYMPTOM_MAP.update(validated_symptom_map_temp)
        # No need for the second verification block, it's redundant now.

        MODEL_KEY_TO_ASK_PHRASE = {details["model_key"]: details["ask_phrase"]
                                   for details in SYMPTOM_MAP.values() if details["model_key"] in MODEL_SYMPTOM_KEYS}
        NATURAL_SYMPTOM_PHRASES_FOR_FUZZY = list(SYMPTOM_MAP.keys())
        app.logger.info(f"SYMPTOM_MAP processed: {len(SYMPTOM_MAP)} natural phrases, {len(MODEL_KEY_TO_ASK_PHRASE)} model key ask phrases.")

    except FileNotFoundError:
        app.logger.error(f"CRITICAL: Model or symptom_columns.pkl not found in {MODEL_DIR}.")
        model = None
    except Exception as e:
        app.logger.error(f"CRITICAL Error initializing model/symptom keys: {e}")
        model = None

    try:
        # Ensure latitude and longitude are numeric, coercing errors to NaN
        doctors_df = pd.read_csv(DOCTORS_CSV_PATH, dtype={'number': str})
        doctors_df.columns = doctors_df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Assuming your CSV has 'latitude' and 'longitude' columns
        # If they have different names (like 'latitude_deg', 'longitude_deg'), adjust here.
        # Also, the CSV header might have spaces, so we use the already cleaned column names.
        # Let's assume after cleaning, the columns are 'latitude' and 'longitude'
        
        # Get the actual cleaned column names for lat/lon
        # Example: if original CSV header was "Latitude (Deg)", it becomes "latitude_(deg)"
        # For your example, it seems they are just "latitude" and "longitude" which is good.
        lat_col_name = 'latitude' # Adjust if your CSV header is different after cleaning
        lon_col_name = 'longitude'# Adjust if your CSV header is different after cleaning

        if lat_col_name in doctors_df.columns:
            doctors_df[lat_col_name] = pd.to_numeric(doctors_df[lat_col_name], errors='coerce')
        else:
            app.logger.warning(f"Latitude column '{lat_col_name}' not found in doctors CSV.")
            doctors_df[lat_col_name] = np.nan # Add an empty column if not present

        if lon_col_name in doctors_df.columns:
            doctors_df[lon_col_name] = pd.to_numeric(doctors_df[lon_col_name], errors='coerce')
        else:
            app.logger.warning(f"Longitude column '{lon_col_name}' not found in doctors CSV.")
            doctors_df[lon_col_name] = np.nan # Add an empty column if not present

        doctors_df['about'] = doctors_df['about'].fillna('N/A')
        doctors_df['image_source'] = doctors_df['image_source'].fillna('https://via.placeholder.com/100?text=No+Image')
        app.logger.info(f"Doctors data loaded from {DOCTORS_CSV_PATH}. Shape: {doctors_df.shape}")
        app.logger.info(f"Doctors DF columns: {doctors_df.columns.tolist()}")
        # Log a few rows with lat/lon to check
        app.logger.info(doctors_df[[doc_name_col if 'doc_name_col' in locals() else 'name', lat_col_name, lon_col_name]].head())


    except Exception as e:
        app.logger.error(f"Error loading doctors data {DOCTORS_CSV_PATH}: {e}")
        doctors_df = pd.DataFrame() # Ensure it's an empty DataFrame on error

    for path, df_name, global_var_name in [
        (DISEASE_DESC_CSV_PATH, "Disease Descriptions", "disease_desc_df"),
        (DISEASE_PRECAUTION_CSV_PATH, "Disease Precautions", "disease_precaution_df")
    ]:
        try:
            temp_df = pd.read_csv(path)
            temp_df.columns = temp_df.columns.str.strip().str.lower()
            if 'disease' in temp_df.columns:
                temp_df.set_index('disease', inplace=True)
                globals()[global_var_name] = temp_df
                app.logger.info(f"{df_name} loaded from {path}")
            else:
                app.logger.error(f"'disease' column not found in {path}")
        except Exception as e:
            app.logger.warning(f"Could not load {df_name} from {path}: {e}")
            globals()[global_var_name] = pd.DataFrame()

initialize_app_data()

# --- Disease to Specialization Map ---
disease_to_specialization_map = {
    # ... (your existing map - keep it as is) ...
    "fungal infection": "Dermatologist", "allergy": "Allergist", "gerd": "Gastroenterologist",
    "chronic cholestasis": "Hepatologist", "drug reaction": "Dermatologist",
    "peptic ulcer diseae": "Gastroenterologist", "aids": "Infectious Disease Specialist",
    "diabetes": "Endocrinologist", "gastroenteritis": "Gastroenterologist",
    "bronchial asthma": "Pulmonologist", "hypertension": "Cardiologist",
    "migraine": "Neurologist", "cervical spondylosis": "Orthopedist",
    "paralysis (brain hemorrhage)": "Neurologist", "jaundice": "Hepatologist",
    "malaria": "Infectious Disease Specialist", "chicken pox": "Dermatologist", # or General Physician
    "dengue": "Infectious Disease Specialist", "typhoid": "Infectious Disease Specialist",
    "hepatitis a": "Hepatologist", "hepatitis b": "Hepatologist", "hepatitis c": "Hepatologist",
    "hepatitis d": "Hepatologist", "hepatitis e": "Hepatologist", "alcoholic hepatitis": "Hepatologist",
    "tuberculosis": "Pulmonologist", "common cold": "General Physician", "pneumonia": "Pulmonologist",
    "dimorphic hemmorhoids(piles)": "Proctologist", "heart attack": "Cardiologist",
    "varicose veins": "Vascular Surgeon", "hypothyroidism": "Endocrinologist",
    "hyperthyroidism": "Endocrinologist", "hypoglycemia": "Endocrinologist",
    "osteoarthristis": "Orthopedist", "arthritis": "Rheumatologist",
    "(vertigo) paroymsal positional vertigo": "ENT Specialist", "acne": "Dermatologist",
    "urinary tract infection": "Urologist", "psoriasis": "Dermatologist", "impetigo": "Dermatologist",
}

def normalize_text(text):
    return str(text).strip().lower() if pd.notna(text) else ""

# --- NLP Symptom Extraction Helper ---
def extract_initial_symptoms_nlp(user_text, symptom_map_config_dict, natural_phrases_list_for_fuzzy, threshold=80):
    # ... (your existing function - keep as is) ...
    identified_model_symptoms = set()
    user_text_lower = user_text.lower()
    if not user_text_lower: return []

    potential_phrases = [p.strip() for p in re.split(r',|\band\b|\bi feel\b|\bi have\b|\bexperiencing\b', user_text_lower) if p.strip()]
    if not potential_phrases: potential_phrases.append(user_text_lower)

    for phrase in potential_phrases:
        if not phrase or len(phrase) < 3: continue # Skip very short phrases
        best_match, score = process.extractOne(phrase, natural_phrases_list_for_fuzzy, scorer=fuzz.token_set_ratio)
        if score >= threshold:
            if best_match in symptom_map_config_dict:
                identified_model_symptoms.add(symptom_map_config_dict[best_match]["model_key"])
    
    app.logger.info(f"NLP Extracted for '{user_text}': {list(identified_model_symptoms)}")
    return list(identified_model_symptoms)


# --- Targeted Symptom Questioning Strategy ---
MIN_SYMPTOMS_FOR_PREDICTION = 4
MAX_QUESTIONS_PER_ROUND = 2

def determine_next_symptoms_to_ask(symptoms_vector_dict, all_model_symptom_keys, ml_model, symptom_map_config_dict, count=MAX_QUESTIONS_PER_ROUND):
    # ... (your existing function - keep as is) ...
    confirmed_positive_keys = [k for k,v in symptoms_vector_dict.items() if v == 1]
    already_addressed_keys = list(symptoms_vector_dict.keys()) # All keys for which we have a 0 or 1

    unasked_symptoms_details = [
        details for natural_phrase, details in symptom_map_config_dict.items()
        if details["model_key"] not in already_addressed_keys or symptoms_vector_dict[details["model_key"]] == 0 # not confirmed positive
                                                              # or simply not addressed yet at all
    ]
    unique_unasked_model_keys = list(set(d["model_key"] for d in unasked_symptoms_details))
    
    if not unique_unasked_model_keys: return []

    import random
    random.shuffle(unique_unasked_model_keys)
    return unique_unasked_model_keys[:count]

# --- In-memory Session Store ---
user_sessions = {}

def get_session(user_id):
    # ... (your existing function - keep as is) ...
    if user_id not in user_sessions:
        app.logger.info(f"New session for user_id: {user_id}")
        user_sessions[user_id] = {
            'state': 'AWAITING_NAME', 'user_name': None,
            'symptoms_vector': {key: 0 for key in MODEL_SYMPTOM_KEYS} if MODEL_SYMPTOM_KEYS else {},
            'symptoms_confirmed_count': 0,
            'symptoms_pending_clarification': [], 'current_clarifying_symptom_key': None,
            'symptoms_targeted_questions_q': [], 'current_targeted_symptom_key': None,
            'age': None, 'sex': None, 'predicted_disease_context': None
        }
    return user_sessions[user_id]

def reset_session_for_new_query(user_id, existing_name=None):
    # ... (your existing function - keep as is) ...
    user_sessions[user_id] = {
        'state': 'AWAITING_INITIAL_SYMPTOMS' if existing_name else 'AWAITING_NAME', # If name known, ask for symptoms
        'user_name': existing_name,
        'symptoms_vector': {key: 0 for key in MODEL_SYMPTOM_KEYS} if MODEL_SYMPTOM_KEYS else {},
        'symptoms_confirmed_count': 0,
        'symptoms_pending_clarification': [], 'current_clarifying_symptom_key': None,
        'symptoms_targeted_questions_q': [], 'current_targeted_symptom_key': None,
        'age': None, 'sex': None, 'predicted_disease_context': None
    }
    return user_sessions[user_id]

# --- Flask Routes ---
@app.route('/')
def chat_home():
    return render_template('chat.html')

@app.route('/chat_api', methods=['POST'])
def chat_api():
    if not model or not MODEL_SYMPTOM_KEYS:
        return jsonify({'bot_response_parts': ["I apologize, my medical knowledge system is currently offline. Please check back soon."]})

    data = request.get_json()
    user_id = data.get('user_id', 'anon_user_' + str(np.random.randint(100000)))
    user_message = data.get('message', '').lower().strip()

    session = get_session(user_id)
    bot_responses = []
    map_data_for_frontend = None # NEW: To hold doctor data for the map
    current_state = session['state']
    user_name_greet = f"{session['user_name']}, " if session['user_name'] else ""

    app.logger.info(f"ID:{user_id}, Name:{session['user_name']}, State:{current_state}, Msg:'{user_message}'")

    # ... (your existing state machine logic for AWAITING_NAME up to AWAITING_SEX)
    # Universal commands
    if user_message in ["reset", "start over"]:
        session = reset_session_for_new_query(user_id) # Resets and gets the new session
        bot_responses.append("Okay, let's start fresh! What's your name?")
        current_state = session['state'] # Update current_state after reset
    elif "help symptoms" in user_message:
        s_list_display = sorted(list(NATURAL_SYMPTOM_PHRASES_FOR_FUZZY))
        bot_responses.append("I understand symptoms related to: " + ", ".join(s_list_display[:10]) + "... and many more. Try describing how you feel.")

    if not bot_responses: # Proceed with state machine if no universal command handled it
        if current_state == 'AWAITING_NAME':
            if user_message and len(user_message) > 1:
                session['user_name'] = user_message.strip().title()
                bot_responses.append(f"Nice to meet you, {session['user_name']}! To help me understand your situation, please describe your main symptoms.")
                session['state'] = 'AWAITING_INITIAL_SYMPTOMS'
            else:
                bot_responses.append("Hello! I'm ArogyaBot. What's your name, please?")
        
        elif current_state == 'AWAITING_INITIAL_SYMPTOMS':
            extracted_keys = extract_initial_symptoms_nlp(user_message, SYMPTOM_MAP, NATURAL_SYMPTOM_PHRASES_FOR_FUZZY)
            newly_identified_keys = [k for k in extracted_keys if session['symptoms_vector'].get(k, 0) == 0 and k != session.get('current_clarifying_symptom_key')]
            
            if newly_identified_keys:
                session['symptoms_pending_clarification'].extend(newly_identified_keys)
                session['symptoms_pending_clarification'] = list(set(session['symptoms_pending_clarification'])) # unique
                
            if session['symptoms_pending_clarification']:
                session['current_clarifying_symptom_key'] = session['symptoms_pending_clarification'].pop(0)
                ask_phrase = MODEL_KEY_TO_ASK_PHRASE.get(session['current_clarifying_symptom_key'], f"Are you experiencing {session['current_clarifying_symptom_key'].replace('_',' ')}?")
                bot_responses.append(user_name_greet + ask_phrase + " (yes/no)")
                session['state'] = 'CLARIFYING_SYMPTOMS'
            elif extracted_keys : # Extracted keys but all were already processed or known
                 bot_responses.append(user_name_greet + "I've noted those. Do you have any other symptoms you'd like to add?")
            else:
                bot_responses.append(user_name_greet + "I'm sorry, I didn't quite catch any symptoms. Could you please describe them again?")

        elif current_state == 'CLARIFYING_SYMPTOMS':
            symptom_key = session['current_clarifying_symptom_key']
            responded = False
            if "yes" in user_message:
                session['symptoms_vector'][symptom_key] = 1
                session['symptoms_confirmed_count'] += 1
                bot_responses.append(f"Noted: {symptom_key.replace('_',' ')}.")
                responded = True
            elif "no" in user_message:
                session['symptoms_vector'][symptom_key] = 0 
                bot_responses.append(f"Okay, no {symptom_key.replace('_',' ')}.")
                responded = True
            
            if responded:
                session['current_clarifying_symptom_key'] = None # Clear current clarification
                if session['symptoms_pending_clarification']: # More from initial NLP
                    session['current_clarifying_symptom_key'] = session['symptoms_pending_clarification'].pop(0)
                    ask_phrase = MODEL_KEY_TO_ASK_PHRASE.get(session['current_clarifying_symptom_key'], f"What about {session['current_clarifying_symptom_key'].replace('_',' ')}?")
                    bot_responses.append(ask_phrase + " (yes/no)")
                else: # No more NLP clarifications, decide next step
                    if session['symptoms_confirmed_count'] < MIN_SYMPTOMS_FOR_PREDICTION:
                        session['symptoms_targeted_questions_q'] = determine_next_symptoms_to_ask(
                            session['symptoms_vector'], MODEL_SYMPTOM_KEYS, model, SYMPTOM_MAP
                        )
                        if session['symptoms_targeted_questions_q']:
                            session['current_targeted_symptom_key'] = session['symptoms_targeted_questions_q'].pop(0)
                            ask_phrase = MODEL_KEY_TO_ASK_PHRASE.get(session['current_targeted_symptom_key'], f"Do you also have {session['current_targeted_symptom_key'].replace('_',' ')}?")
                            bot_responses.append(ask_phrase + " (yes/no)")
                            session['state'] = 'TARGETED_QUESTIONING'
                        else: # Not enough symptoms and no more to ask from list
                            bot_responses.append(user_name_greet + "I need a bit more information. Are there any other symptoms at all, even minor ones?")
                            session['state'] = 'AWAITING_INITIAL_SYMPTOMS' # Go back to gather more
                    else:
                        bot_responses.append(user_name_greet + "Thanks. For our records, what is your age?")
                        session['state'] = 'AWAITING_AGE'
            else: # Invalid yes/no response
                ask_phrase = MODEL_KEY_TO_ASK_PHRASE.get(symptom_key, f"Are you experiencing {symptom_key.replace('_',' ')}?")
                bot_responses.append("Please answer 'yes' or 'no'. " + ask_phrase)


        elif current_state == 'TARGETED_QUESTIONING':
            symptom_key = session['current_targeted_symptom_key']
            responded = False
            if "yes" in user_message:
                session['symptoms_vector'][symptom_key] = 1
                session['symptoms_confirmed_count'] += 1
                bot_responses.append(f"Understood: {symptom_key.replace('_',' ')}.")
                responded = True
            elif "no" in user_message:
                session['symptoms_vector'][symptom_key] = 0
                bot_responses.append(f"Okay, no {symptom_key.replace('_',' ')}.")
                responded = True

            if responded:
                session['current_targeted_symptom_key'] = None
                # Ask one more if available & still below min+1 (or some other threshold)
                if session['symptoms_targeted_questions_q'] and session['symptoms_confirmed_count'] < (MIN_SYMPTOMS_FOR_PREDICTION + 1): 
                    session['current_targeted_symptom_key'] = session['symptoms_targeted_questions_q'].pop(0)
                    ask_phrase = MODEL_KEY_TO_ASK_PHRASE.get(session['current_targeted_symptom_key'], f"And how about {session['current_targeted_symptom_key'].replace('_',' ')}?")
                    bot_responses.append(ask_phrase + " (yes/no)")
                elif session['symptoms_confirmed_count'] < MIN_SYMPTOMS_FOR_PREDICTION and not session['symptoms_targeted_questions_q']:
                    # Still not enough symptoms, and ran out of targeted questions
                    bot_responses.append(user_name_greet + "I still need a bit more information. Can you think of any other symptoms you're experiencing?")
                    session['state'] = 'AWAITING_INITIAL_SYMPTOMS' # Go back to general symptom gathering
                else: # Either enough symptoms or ran out of questions but met minimum
                    bot_responses.append(user_name_greet + "Thank you. Just a couple of routine questions. What is your age?")
                    session['state'] = 'AWAITING_AGE'
            else: # Invalid yes/no
                ask_phrase = MODEL_KEY_TO_ASK_PHRASE.get(symptom_key, f"Do you have {symptom_key.replace('_',' ')}?")
                bot_responses.append("Please answer 'yes' or 'no'. " + ask_phrase)

        elif current_state == 'AWAITING_AGE':
            try:
                age_match = re.search(r'\d+', user_message)
                if age_match:
                    age = int(age_match.group(0))
                    if 0 < age < 120:
                        session['age'] = age
                        bot_responses.append(user_name_greet + f"Age {age} noted. And your biological sex? (Male/Female/Prefer not to say)")
                        session['state'] = 'AWAITING_SEX'
                    else: bot_responses.append("That age seems unlikely. Could you please provide a valid age?")
                else: bot_responses.append("I couldn't understand the age. Please enter it as a number (e.g., 'I am 30').")
            except ValueError: bot_responses.append("Please enter your age using numbers.")

        elif current_state == 'AWAITING_SEX':
            sex_val = None
            if "male" in user_message or user_message == "m": sex_val = "Male"
            elif "female" in user_message or user_message == "f": sex_val = "Female"
            elif "other" in user_message or "prefer not to say" in user_message or "skip" in user_message or "n/a" in user_message : sex_val = "Prefer not to say"
            
            if sex_val:
                session['sex'] = sex_val
                bot_responses.append(user_name_greet + f"{sex_val} recorded. Thank you. I'll analyze your symptoms now...")
                session['state'] = 'READY_TO_PREDICT' # Transition to prediction
            else:
                bot_responses.append(user_name_greet + "Please specify Male, Female, or you can say 'Prefer not to say'.")
        
        # PREDICTION STATE (can be entered directly if enough symptoms and demographics gathered)
        if session['state'] == 'READY_TO_PREDICT': # Note: This check should be AFTER age/sex if they are mandatory
            if session['symptoms_confirmed_count'] < 1 : # Or your MIN_SYMPTOMS_FOR_PREDICTION
                bot_responses.append(user_name_greet + "I don't seem to have enough symptom information to make an analysis. Could we start over by you telling me your main symptoms?")
                session['state'] = 'AWAITING_INITIAL_SYMPTOMS'
            elif session['age'] is None or session['sex'] is None: # Ensure demographics are gathered
                if session['age'] is None:
                    bot_responses.append(user_name_greet + "Before I proceed, what is your age?")
                    session['state'] = 'AWAITING_AGE'
                elif session['sex'] is None:
                    bot_responses.append(user_name_greet + "And your biological sex? (Male/Female/Prefer not to say)")
                    session['state'] = 'AWAITING_SEX'
            else:
                app.logger.info(f"Predicting for vector: {session['symptoms_vector']}")
                input_df = pd.DataFrame([session['symptoms_vector']])
                input_df = input_df[MODEL_SYMPTOM_KEYS] # Ensure correct column order
                
                pred_proba = model.predict_proba(input_df)[0]
                pred_idx = np.argmax(pred_proba)
                disease_raw = model.classes_[pred_idx]
                confidence = pred_proba[pred_idx] * 100
                session['predicted_disease_context'] = disease_raw
                disease_clean = disease_raw.strip().title()

                bot_responses.append(user_name_greet + f"based on the symptoms, my analysis suggests it might be **{disease_clean}** (Confidence: {confidence:.2f}%).")
                
                norm_disease_key = normalize_text(disease_raw)
                description = "No detailed description available for this condition."
                if not disease_desc_df.empty and norm_disease_key in disease_desc_df.index:
                    description = disease_desc_df.loc[norm_disease_key, 'description']
                
                precautions = []
                if not disease_precaution_df.empty and norm_disease_key in disease_precaution_df.index:
                    prec_series = disease_precaution_df.loc[norm_disease_key]
                    precautions = [prec_series[f'precaution_{i}'] for i in range(1,5) if pd.notna(prec_series.get(f'precaution_{i}'))]

                bot_responses.append(f"*{description}*")
                if precautions: bot_responses.append("\n**Recommended Precautions:**\n- " + "\n- ".join(precautions))
                bot_responses.append("\n**Important Disclaimer:** This is an AI-generated suggestion and not a substitute for professional medical diagnosis. Please consult a qualified doctor for accurate advice and treatment.")
                bot_responses.append(f"\nWould you like me to look for doctors in Bangladesh who might treat **{disease_clean}**, {session['user_name']}? (yes/no)")
                session['state'] = 'AWAITING_DOCTOR_CONFIRMATION'

        elif current_state == 'AWAITING_DOCTOR_CONFIRMATION':
            disease_context = session['predicted_disease_context']
            responded_to_doc_q = False
            doctors_for_map_list = [] # NEW: List to hold doctor data for mapping

            if "yes" in user_message and disease_context:
                norm_disease_key = normalize_text(disease_context)
                target_spec_from_map = disease_to_specialization_map.get(norm_disease_key)
                
                found_docs_messages = [f"I'm sorry, {session['user_name']}, I couldn't immediately find doctors specifically listed for '{disease_context.title()}' or its related specialty in my current database."]
                
                if target_spec_from_map and not doctors_df.empty:
                    relevant_docs_df = doctors_df[
                        doctors_df['speciality'].str.contains(target_spec_from_map, case=False, na=False) &
                        doctors_df['latitude'].notna() &  # Ensure lat/lon are present
                        doctors_df['longitude'].notna()
                    ]
                    
                    if not relevant_docs_df.empty:
                        found_docs_messages = [f"For a condition like **{disease_context.title()}**, you would typically consult a **{target_spec_from_map}**. Here are a few doctors listed with that or a similar specialty in Bangladesh. I can also show them on a map."]
                        
                        # Get column names dynamically after cleaning
                        name_col = 'name' # default, adjust if your actual cleaned name is different
                        spec_col = 'speciality'
                        hosp_col = 'hospital_name'
                        addr_col = 'address'
                        num_col = 'number'
                        img_col = 'image_source'
                        lat_col = 'latitude'
                        lon_col = 'longitude'


                        for i, (_, doc) in enumerate(relevant_docs_df.head(3).iterrows()):
                            doc_name = doc.get(name_col, 'N/A')
                            doc_spec = doc.get(spec_col, 'N/A')
                            doc_hosp = doc.get(hosp_col, 'N/A')
                            doc_addr = doc.get(addr_col, 'N/A')
                            doc_contact = doc.get(num_col, 'N/A')
                            doc_img = doc.get(img_col, 'https_via.placeholder.com/80')
                            doc_lat = doc.get(lat_col) # Already numeric or NaN
                            doc_lon = doc.get(lon_col) # Already numeric or NaN

                            doc_info_html = (f"<div style='border:1px solid #eee; padding:10px; margin-bottom:10px; border-radius:5px;'>"
                                        f"<img src='{doc_img}' alt='{doc_name}' style='width:60px; height:60px; border-radius:50%; float:left; margin-right:10px; object-fit:cover;'>"
                                        f"<strong>{i+1}. {doc_name}</strong><br>"
                                        f"<em>{doc_spec}</em><br>"
                                        f"🏥 {doc_hosp}<br>"
                                        f"📍 <small>{doc_addr}</small><br>"
                                        f"{'📞 '+str(doc_contact) if pd.notna(doc_contact) else ''}" # Ensure contact is string
                                        f"<div style='clear:both;'></div></div>")
                            found_docs_messages.append(doc_info_html)

                            # Add to list for mapping if lat/lon are valid
                            if pd.notna(doc_lat) and pd.notna(doc_lon):
                                doctors_for_map_list.append({
                                    "name": doc_name,
                                    "speciality": doc_spec,
                                    "hospital": doc_hosp,
                                    "address": doc_addr,
                                    "contact": str(doc_contact) if pd.notna(doc_contact) else 'N/A',
                                    "lat": doc_lat,
                                    "lng": doc_lon,
                                    "image": doc_img
                                })
                        
                        if doctors_for_map_list:
                             map_data_for_frontend = {"doctors": doctors_for_map_list} # Prepare for JSON
                             found_docs_messages.append("Check the map display for their locations.")
                        else:
                            found_docs_messages.append("I found some doctors, but unfortunately, I don't have location data for them to display on a map.")
                        
                        found_docs_messages.append("It's always best to call ahead to confirm availability and suitability.")
                    else:
                        found_docs_messages = [f"While a **{target_spec_from_map}** would be suitable for **{disease_context.title()}**, I don't have specific doctors listed under that exact specialty title with location data in my current BD database. You may need to search more broadly or consult a general physician for a referral."]

                bot_responses.extend(found_docs_messages)
                responded_to_doc_q = True
            elif "no" in user_message:
                bot_responses.append(f"Alright, {session['user_name']}. Please take good care of yourself.")
                responded_to_doc_q = True
            
            if responded_to_doc_q:
                bot_responses.append("\nIs there anything else I can help you with today?")
                session = reset_session_for_new_query(user_id, session['user_name'])
                current_state = session['state']
            else:
                bot_responses.append("Please answer 'yes' or 'no' regarding the doctor search.")

    if not bot_responses:
        greeting = f"{session.get('user_name', 'there')}, " if session.get('user_name') else ""
        bot_responses.append(f"I'm sorry, {greeting}I'm not sure how to respond to that. You can tell me your symptoms, or type 'reset' to start over.")

    app.logger.info(f"BOT for {user_id} (Name: {session.get('user_name')}): {bot_responses}")
    
    # Construct the final JSON response
    json_response = {'bot_response_parts': bot_responses, 'user_id': user_id}
    if map_data_for_frontend:
        json_response['map_data'] = map_data_for_frontend # Add map data if available
        app.logger.info(f"Sending map data: {map_data_for_frontend}")


    return jsonify(json_response)


if __name__ == '__main__':
    if not model or not MODEL_SYMPTOM_KEYS:
        print("="*80)
        print("ERROR: MODEL OR SYMPTOM KEYS NOT LOADED. FLASK APP CANNOT RUN CORRECTLY.")
        # ... (rest of your error messages)
        print("="*80)

    else:
        app.logger.info("Flask app starting... Model and initial data loaded.")
        app.run(debug=True, port=5002, use_reloader=True)