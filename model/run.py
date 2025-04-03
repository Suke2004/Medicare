import numpy as np
import re
import joblib as jb
import speech_recognition as sr
import pandas as pd

Audio = sr.Recognizer()

def recognize_speech():
    with sr.Microphone() as source:
        print("Hey, tell me your problem....")
        print('\n')
        audio = Audio.listen(source)
    try:
        text = Audio.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

model = jb.load(r'C:\MedRecommendation-main\MedRecommendation\diseasepred.pkl')

symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 
            'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 
            'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 
            'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 
            'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 
            'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 
            'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 
            'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 
            'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 
            'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 
            'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 
            'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 
            'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 
            'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 
            'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 
            'bladder_discomfort', 'foul_smell_of_urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 
            'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 
            'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 
            'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 
            'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 
            'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 
            'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 
            'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 
            'red_sore_around_nose', 'yellow_crust_ooze']

for i in range(len(symptoms)):
    if '_' in symptoms[i]:
        symptoms[i] = symptoms[i].replace('_', ' ')
        
disease_dict = {'(vertigo) Paroymsal Positional Vertigo': 0, 'AIDS': 1, 'Acne': 2, 'Alcoholic hepatitis': 3, 'Allergy': 4, 
                'Arthritis': 5, 'Bronchial Asthma': 6, 'Cervical spondylosis': 7, 'Chicken pox': 8, 'Chronic cholestasis': 9, 
                'Common Cold': 10, 'Dengue': 11, 'Diabetes': 12, 'Dimorphic hemmorhoids(piles)': 13, 'Drug Reaction': 14, 
                'Fungal infection': 15, 'GERD': 16, 'Gastroenteritis': 17, 'Heart attack': 18, 'Hepatitis B': 19, 
                'Hepatitis C': 20, 'Hepatitis D': 21, 'Hepatitis E': 22, 'Hypertension': 23, 'Hyperthyroidism': 24, 
                'Hypoglycemia': 25, 'Hypothyroidism': 26, 'Impetigo': 27, 'Jaundice': 28, 'Malaria': 29, 'Migraine': 30, 
                'Osteoarthristis': 31, 'Paralysis (brain hemorrhage)': 32, 'Peptic ulcer disease': 33, 'Pneumonia': 34, 
                'Psoriasis': 35, 'Tuberculosis': 36, 'Typhoid': 37, 'Urinary tract infection': 38, 'Varicose veins': 39, 
                'Hepatitis A': 40}

description_dict = {
    '(vertigo) Paroymsal Positional Vertigo': 'Paroxysmal Positional Vertigo (PPV) is a common cause of dizziness. It is characterized by brief episodes of mild to intense dizziness associated with specific changes in the position of your head, such as tipping your head up or down. Symptoms include vertigo, dizziness, imbalance, nausea, and vomiting. These symptoms are caused by changes in head position and usually last less than a minute. The condition is usually not serious and can be treated with specific head and body movements performed by a healthcare professional.',
    'AIDS': 'Acquired Immunodeficiency Syndrome (AIDS) is a chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV). HIV damages the immune system, interfering with the body’s ability to fight infections and diseases. Symptoms include rapid weight loss, recurring fever, extreme tiredness, prolonged swelling of the lymph glands, and sores of the mouth, anus, or genitals. Without treatment, HIV can lead to AIDS, where the immune system is severely compromised, making the body vulnerable to opportunistic infections and certain cancers.',
    'Acne': 'Acne is a skin condition that occurs when hair follicles become plugged with oil and dead skin cells. It often causes whiteheads, blackheads, or pimples, and usually appears on the face, forehead, chest, upper back, and shoulders. Acne is most common among teenagers, though it affects people of all ages. Symptoms include inflamed papules, pustules, nodules, and cystic lesions. Factors that can worsen acne include hormones, certain medications, diet, and stress. Treatments range from topical and oral medications to lifestyle changes and skincare routines.',
    'Alcoholic hepatitis': 'Alcoholic hepatitis is an inflammation of the liver caused by excessive alcohol consumption. It can range from mild to severe and can be life-threatening. Symptoms include jaundice (yellowing of the skin and eyes), loss of appetite, nausea, vomiting, abdominal pain, and swelling. Other symptoms may include fever, fatigue, and weight loss. The condition can lead to liver failure if alcohol consumption continues. Treatment involves stopping alcohol intake, nutritional support, and medications to reduce inflammation and manage symptoms.',
    'Allergy': 'Allergies occur when the immune system reacts to a foreign substance, such as pollen, bee venom, pet dander, or a particular food. Symptoms vary depending on the allergen and can include sneezing, itching, runny nose, hives, swelling, and in severe cases, anaphylaxis, a life-threatening reaction. Common allergens include pollen, dust mites, mold, animal dander, and certain foods. Treatment includes avoiding known allergens, medications to relieve symptoms, and immunotherapy (allergy shots) for long-term relief.',
    'Arthritis': 'Arthritis is inflammation of one or more joints, causing pain and stiffness that can worsen with age. The most common types are osteoarthritis and rheumatoid arthritis. Symptoms include joint pain, stiffness, swelling, redness, and decreased range of motion. Osteoarthritis involves wear-and-tear damage to joint cartilage, while rheumatoid arthritis is an autoimmune disorder that targets joint linings. Treatments include medications, physical therapy, lifestyle changes, and sometimes surgery to reduce symptoms and improve joint function.',
    'Bronchial Asthma': 'Bronchial asthma is a chronic inflammatory disease of the airways that causes periodic attacks of coughing, shortness of breath, chest tightness, and wheezing. Symptoms can be triggered by allergens, respiratory infections, cold air, exercise, and stress. Asthma attacks vary in severity and can be life-threatening if not properly managed. Treatment includes avoiding triggers, using inhalers to manage symptoms, and medications to control inflammation. Long-term control involves identifying and avoiding triggers and regular monitoring.',
    'Cervical spondylosis': 'Cervical spondylosis is age-related wear and tear affecting the spinal disks in your neck. Symptoms include neck pain and stiffness, headache, pain in the shoulder or arms, and, in severe cases, loss of balance and bladder control. The condition is caused by the degeneration of cartilage and bones, leading to the formation of bone spurs. Treatments include medications, physical therapy, and sometimes surgery to relieve pain and prevent permanent spinal cord or nerve damage.',
    'Chicken pox': 'Chickenpox is a highly contagious viral infection causing an itchy, blister-like rash. It is caused by the varicella-zoster virus. Symptoms include a red, itchy rash that usually starts on the face, chest, and back, and then spreads to the rest of the body. Other symptoms include fever, loss of appetite, headache, and tiredness. The disease is usually mild, but can be serious, especially in babies, adults, and people with weakened immune systems. A vaccine is available for prevention.',
    'Chronic cholestasis': 'Chronic cholestasis is a condition where bile cannot flow from the liver to the duodenum. Symptoms include jaundice (yellowing of the skin and eyes), dark urine, pale stools, itching, and fatigue. Causes can include liver diseases, bile duct obstruction, and genetic disorders. Treatment depends on the underlying cause and may include medications, dietary changes, or surgery to relieve bile duct obstruction and improve bile flow.',
    'Common Cold': 'The common cold is a viral infection of the nose and throat. Symptoms include a runny or stuffy nose, sore throat, cough, congestion, slight body aches, sneezing, low-grade fever, and general malaise. It is usually harmless, and most people recover within a week or ten days. More than 200 viruses can cause a cold, but rhinoviruses are the most common culprit. Treatment focuses on relieving symptoms, including rest, fluids, and over-the-counter medications.',
    'Dengue': 'Dengue fever is a mosquito-borne viral infection causing flu-like illness. Symptoms include high fever, severe headache, pain behind the eyes, joint and muscle pain, rash, and mild bleeding (such as nose or gum bleed). Severe dengue can cause plasma leakage, fluid accumulation, respiratory distress, severe bleeding, and organ impairment. There is no specific treatment for dengue, but early detection and proper medical care lower fatality rates. Prevention involves avoiding mosquito bites and reducing mosquito habitats.',
    'Diabetes': 'Diabetes is a chronic disease that affects how your body turns food into energy. There are three main types: Type 1, Type 2, and gestational diabetes. Symptoms include increased thirst, frequent urination, extreme hunger, unintended weight loss, fatigue, and blurred vision. Long-term complications can include heart disease, stroke, kidney disease, eye problems, and nerve damage. Management includes monitoring blood sugar levels, following a healthy diet, exercising regularly, and taking medications or insulin therapy.',
    'Dimorphic hemmorhoids(piles)': 'Hemorrhoids, also known as piles, are swollen veins in the lower rectum and anus. Symptoms include painless bleeding during bowel movements, itching or irritation in the anal region, pain or discomfort, swelling around the anus, and a lump near the anus, which may be sensitive or painful. Hemorrhoids can be internal or external. Causes include straining during bowel movements, sitting for long periods, and chronic constipation or diarrhea. Treatments range from lifestyle changes and over-the-counter remedies to medical procedures.',
    'Drug Reaction': 'A drug reaction is an adverse response to medication. Symptoms can vary widely depending on the drug and individual, ranging from mild reactions like rashes, itching, and gastrointestinal upset to severe reactions such as anaphylaxis, organ damage, and Stevens-Johnson syndrome. Treatment involves discontinuing the offending drug, managing symptoms with medications like antihistamines or corticosteroids, and, in severe cases, providing emergency medical care. Identifying and avoiding the drug in the future is crucial.',
    'Fungal infection': 'Fungal infections are caused by fungi and can affect various parts of the body, including the skin, nails, and respiratory system. Common symptoms include itching, redness, rash, blisters, and scaling. Infections can range from mild (like athlete’s foot) to severe (like invasive aspergillosis). Treatment depends on the type and severity of the infection and may include topical or oral antifungal medications. Preventive measures include keeping the affected area clean and dry and avoiding sharing personal items.',
    'GERD': 'Gastroesophageal reflux disease (GERD) is a chronic digestive condition where stomach acid or bile irritates the food pipe lining. Symptoms include heartburn, regurgitation, chest pain, difficulty swallowing, and a feeling of a lump in the throat. Long-term complications can include esophagitis, Barrett’s esophagus, and an increased risk of esophageal cancer. Treatment involves lifestyle changes, medications to reduce stomach acid, and in severe cases, surgery. Avoiding trigger foods and eating smaller meals can help manage symptoms.',
    'Gastroenteritis': 'Gastroenteritis is an inflammation of the stomach and intestines, typically resulting from a viral or bacterial infection. Symptoms include diarrhea, vomiting, abdominal cramps, nausea, and sometimes fever. Dehydration is a common complication, especially in young children and older adults. The condition is usually self-limiting and resolves within a few days. Treatment focuses on staying hydrated, rest, and, in some cases, medications to alleviate symptoms. Good hygiene practices can help prevent the spread of gastroenteritis.',
    'Heart attack': 'A heart attack, or myocardial infarction, occurs when blood flow to a part of the heart is blocked for a long enough time to cause damage or death to the heart muscle. Symptoms include chest pain or discomfort, shortness of breath, nausea, lightheadedness, and pain in the shoulders, arms, neck, or jaw. Immediate medical attention is crucial. Treatment involves medications, lifestyle changes, and procedures like angioplasty or surgery to restore blood flow and prevent further damage.',
    'Hepatitis B': 'Hepatitis B is a viral infection that attacks the liver and can cause both acute and chronic disease. Symptoms include jaundice, dark urine, extreme fatigue, nausea, vomiting, and abdominal pain. Chronic infection can lead to liver cirrhosis and liver cancer. The virus is transmitted through contact with infectious body fluids, such as blood, semen, and vaginal fluids. Vaccination is the most effective way to prevent hepatitis B. Treatment for chronic infection involves antiviral medications to reduce liver damage.',
    'Hepatitis C': 'Hepatitis C is a viral infection that causes liver inflammation, sometimes leading to serious liver damage. Many people with hepatitis C have no symptoms until the virus causes liver damage. Symptoms include fatigue, fever, muscle and joint pain, nausea, poor appetite, jaundice, and dark urine. It is transmitted through blood, often via sharing needles or other equipment to inject drugs. Antiviral medications can cure most cases of hepatitis C. Early diagnosis and treatment can prevent long-term liver damage.',
    'Hepatitis D': 'Hepatitis D is a liver infection caused by the hepatitis D virus (HDV) and occurs only in people who are infected with hepatitis B. Symptoms are similar to those of other forms of hepatitis, including jaundice, fatigue, abdominal pain, nausea, and vomiting. Co-infection with hepatitis B can lead to more severe liver disease and a higher risk of liver cirrhosis and liver cancer. There is no specific vaccine for hepatitis D, but hepatitis B vaccination can prevent HDV infection.',
    'Hepatitis E': 'Hepatitis E is a viral liver disease that usually results in a self-limiting infection. Symptoms include jaundice, fatigue, nausea, vomiting, abdominal pain, and dark urine. The disease is primarily spread through contaminated drinking water. In most cases, it resolves on its own within a few weeks, but it can be severe in pregnant women and people with weakened immune systems. There is no specific treatment for hepatitis E, but preventive measures include good sanitation and access to clean drinking water.',
    'Hypertension': 'Hypertension, or high blood pressure, is a condition where the force of the blood against the artery walls is too high. It often has no symptoms, but long-term high blood pressure can lead to serious health problems, including heart disease, stroke, and kidney disease. Symptoms, if present, may include headaches, shortness of breath, and nosebleeds. Management includes lifestyle changes such as diet and exercise, and medications to lower blood pressure. Regular monitoring is essential for controlling hypertension.',
    'Hyperthyroidism': 'Hyperthyroidism is a condition where the thyroid gland produces too much thyroid hormone. Symptoms include weight loss, rapid heartbeat, increased appetite, nervousness, sweating, and irritability. Other symptoms can include tremors, fatigue, muscle weakness, and difficulty sleeping. Causes include Graves disease, thyroid nodules, and thyroiditis. Treatment options include medications, radioactive iodine therapy, and surgery. Managing hyperthyroidism involves regular monitoring and treatment to control thyroid hormone levels and alleviate symptoms.',
    'Hypoglycemia': 'Hypoglycemia is a condition characterized by abnormally low blood sugar levels. Symptoms include shakiness, sweating, confusion, rapid heartbeat, dizziness, hunger, and irritability. Severe hypoglycemia can lead to seizures, unconsciousness, and even death if not treated promptly. Causes can include excessive insulin use, certain medications, prolonged fasting, and excessive alcohol consumption. Treatment involves consuming fast-acting carbohydrates, such as glucose tablets or juice, to raise blood sugar levels quickly and addressing the underlying cause to prevent recurrence.',
    'Hypothyroidism': 'Hypothyroidism is a condition where the thyroid gland does not produce enough thyroid hormones. Symptoms include fatigue, weight gain, cold intolerance, constipation, dry skin, hair loss, muscle weakness, depression, and slowed heart rate. It is commonly caused by autoimmune thyroiditis (Hashimotos disease), radiation treatment, or surgical removal of the thyroid gland. Treatment involves daily use of synthetic thyroid hormone levothyroxine, which normalizes hormone levels and alleviates symptoms. Regular monitoring and dosage adjustments are necessary.',
    'Impetigo': 'Impetigo is a highly contagious bacterial skin infection common in young children. It typically appears as red sores on the face, especially around the nose and mouth, and on hands and feet. The sores burst and develop honey-colored crusts. Other symptoms can include itching and mild discomfort. The condition is caused by Staphylococcus aureus or Streptococcus pyogenes bacteria. Treatment involves topical or oral antibiotics to clear the infection and measures to prevent spreading the infection to others.',
    'Jaundice': 'Jaundice is a condition where the skin, whites of the eyes, and mucous membranes turn yellow due to a high level of bilirubin, a yellow-orange bile pigment. Symptoms include yellowing of the skin and eyes, dark urine, and pale stools. Causes can include liver diseases, bile duct obstruction, hemolytic anemia, and infections. Treatment depends on the underlying cause and may involve medication, surgery, or other medical interventions to lower bilirubin levels and treat the root cause.',
    'Malaria': 'Malaria is a mosquito-borne infectious disease caused by Plasmodium parasites. Symptoms include fever, chills, headache, muscle pain, and fatigue. Other symptoms can include nausea, vomiting, and diarrhea. Severe malaria can cause anemia, respiratory distress, organ failure, and death if not treated promptly. It is prevalent in tropical and subtropical regions. Treatment involves antimalarial medications, and prevention includes using mosquito nets, repellents, and prophylactic drugs. Early diagnosis and treatment are crucial for recovery and preventing complications.',
    'Migraine': 'Migraine is a neurological disorder characterized by recurrent headaches that are moderate to severe. Symptoms include throbbing pain, usually on one side of the head, nausea, vomiting, and sensitivity to light and sound. Migraines can last for hours to days and may be preceded by warning symptoms called aura, such as visual disturbances. Triggers can include stress, certain foods, hormonal changes, and environmental factors. Treatment involves medications to relieve symptoms and preventive measures to reduce the frequency and severity of attacks.',
    'Osteoarthritis': 'Osteoarthritis is a degenerative joint disease characterized by the breakdown of cartilage that cushions the ends of the bones. Symptoms include joint pain, stiffness, swelling, and decreased range of motion. It commonly affects the knees, hips, hands, and spine. Risk factors include aging, joint injury, obesity, and genetics. Treatment focuses on relieving symptoms and improving joint function, including medications, physical therapy, lifestyle changes, and in severe cases, joint replacement surgery. Regular exercise and maintaining a healthy weight can help manage symptoms.',
    'Paralysis (brain hemorrhage)': 'Paralysis due to a brain hemorrhage occurs when bleeding in the brain damages neural pathways, leading to loss of muscle function in part of the body. Symptoms include sudden weakness or numbness, especially on one side of the body, difficulty speaking, vision problems, severe headache, and loss of coordination. Causes can include high blood pressure, head injury, aneurysms, and blood vessel abnormalities. Immediate medical treatment is crucial to minimize brain damage. Rehabilitation involves physical therapy to regain as much function as possible.',
    'Peptic ulcer disease': 'Peptic ulcer disease (PUD) involves sores or ulcers developing in the lining of the stomach, lower esophagus, or small intestine. Symptoms include burning stomach pain, bloating, heartburn, nausea, and intolerance to fatty foods. The most common causes are infection with Helicobacter pylori bacteria and long-term use of NSAIDs. Treatment includes medications to reduce stomach acid, antibiotics to treat H. pylori infection, and lifestyle changes. Avoiding spicy foods, alcohol, and smoking can help manage symptoms and prevent recurrence.',
    'Pneumonia': 'Pneumonia is an infection that inflames the air sacs in one or both lungs. Symptoms include chest pain, coughing (with phlegm), fever, chills, and difficulty breathing. It can be caused by bacteria, viruses, or fungi. Risk factors include age, smoking, chronic illness, and weakened immune systems. Treatment depends on the cause and may include antibiotics, antiviral drugs, or antifungal medications, along with supportive care such as rest, fluids, and oxygen therapy. Vaccination can help prevent some types of pneumonia.',
    'Psoriasis': 'Psoriasis is a chronic autoimmune condition that causes the rapid buildup of skin cells, leading to scaling on the skin’s surface. Symptoms include red patches of skin covered with thick, silvery scales, dry, cracked skin that may bleed, itching, burning, soreness, and swollen or stiff joints. It can occur anywhere on the body but commonly affects the scalp, elbows, knees, and lower back. Treatment aims to remove scales and stop skin cells from growing quickly and may include topical treatments, phototherapy, and systemic medications.',
    'Tuberculosis': 'Tuberculosis (TB) is a contagious bacterial infection caused by Mycobacterium tuberculosis. It primarily affects the lungs but can spread to other organs. Symptoms include a persistent cough (lasting more than three weeks), chest pain, coughing up blood, fatigue, weight loss, fever, night sweats, and loss of appetite. TB is spread through the air when an infected person coughs or sneezes. Treatment involves a long course of antibiotics, usually lasting six to nine months. Early detection and adherence to the treatment regimen are crucial for curing TB and preventing its spread.',
    'Typhoid': 'Typhoid fever is a bacterial infection caused by Salmonella typhi. Symptoms include high fever, weakness, stomach pain, headache, diarrhea or constipation, and a rash. It is typically spread through contaminated food and water. Severe cases can lead to complications such as intestinal bleeding or perforation. Treatment involves antibiotics to kill the bacteria, and supportive care to manage symptoms. Vaccination and good sanitation practices are important preventive measures. Prompt medical treatment is essential for recovery and preventing complications.',
    'Urinary tract infection': 'A urinary tract infection (UTI) is an infection in any part of the urinary system, including kidneys, bladder, ureters, and urethra. Symptoms include a strong, persistent urge to urinate, a burning sensation when urinating, cloudy urine, urine with a strong odor, and pelvic pain (in women). UTIs are more common in women than men. Causes include bacterial infection, sexual activity, and poor hygiene. Treatment typically involves antibiotics to clear the infection and drinking plenty of fluids to help flush bacteria from the urinary tract.',
    'Varicose veins': 'Varicose veins are enlarged, twisted veins that usually occur in the legs and feet. Symptoms include aching pain, swelling, heaviness, and discomfort in the legs, as well as visible, swollen veins. Causes include standing or sitting for long periods, age, pregnancy, and obesity. Treatment options range from lifestyle changes, such as exercise and wearing compression stockings, to medical procedures like sclerotherapy, laser treatments, and surgery to remove or close off the affected veins.',
    'Hepatitis A': 'Hepatitis A is a highly contagious liver infection caused by the hepatitis A virus. Symptoms include fatigue, nausea, abdominal pain, loss of appetite, low-grade fever, and jaundice. It spreads through ingestion of contaminated food and water or close contact with an infected person. Most people recover fully with supportive care, and the disease does not become chronic. Vaccination is effective for prevention, and practicing good hygiene, including handwashing, can reduce the risk of infection.'
}

medications_description = {
    "['Antifungal Cream', 'Fluconazole', 'Terbinafine', 'Clotrimazole', 'Ketoconazole']": {
        "Antifungal Cream": "Topical medication used to treat fungal infections on the skin by inhibiting fungal growth.",
        "Fluconazole": "Oral antifungal medication used to treat various fungal infections by disrupting the fungal cell membrane.",
        "Terbinafine": "Oral and topical antifungal medication that works by inhibiting an enzyme in fungi, leading to cell death.",
        "Clotrimazole": "Topical antifungal cream used to treat skin infections by stopping the growth of fungi.",
        "Ketoconazole": "Antifungal medication available in oral and topical forms to treat infections by inhibiting fungal cell membrane formation."
    },
    "['Antihistamines', 'Decongestants', 'Epinephrine', 'Corticosteroids', 'Immunotherapy']": {
        "Antihistamines": "Medications that relieve allergy symptoms by blocking histamine receptors, reducing itching, sneezing, and runny nose.",
        "Decongestants": "Medications that relieve nasal congestion by shrinking swollen blood vessels in the nasal passages.",
        "Epinephrine": "Emergency medication used to treat severe allergic reactions (anaphylaxis) by relaxing muscles in the airways and tightening blood vessels.",
        "Corticosteroids": "Anti-inflammatory medications that reduce swelling and suppress the immune response in various conditions.",
        "Immunotherapy": "Treatment that enhances or suppresses the immune system to help the body fight diseases, including allergies and cancers."
    },
    "['Proton Pump Inhibitors (PPIs)', 'H2 Blockers', 'Antacids', 'Prokinetics', 'Antibiotics']": {
        "Proton Pump Inhibitors (PPIs)": "Medications that reduce stomach acid production by blocking the enzyme in the stomach wall that produces acid.",
        "H2 Blockers": "Medications that reduce stomach acid by blocking histamine receptors in the stomach lining.",
        "Antacids": "Over-the-counter medications that neutralize stomach acid to relieve heartburn and indigestion.",
        "Prokinetics": "Medications that enhance gastrointestinal motility by increasing the movement of the digestive tract.",
        "Antibiotics": "Medications used to treat bacterial infections by killing or inhibiting the growth of bacteria."
    },
    "['Ursodeoxycholic acid', 'Cholestyramine', 'Methotrexate', 'Corticosteroids', 'Liver transplant']": {
        "Ursodeoxycholic acid": "Medication used to dissolve gallstones and treat certain liver diseases by reducing cholesterol production in the liver.",
        "Cholestyramine": "Medication that binds bile acids in the intestine to lower cholesterol levels and relieve itching caused by liver disease.",
        "Methotrexate": "Immunosuppressant medication used to treat cancer, autoimmune diseases, and severe psoriasis by inhibiting cell growth.",
        "Corticosteroids": "Anti-inflammatory medications that reduce swelling and suppress the immune response in various conditions.",
        "Liver transplant": "Surgical procedure to replace a diseased liver with a healthy one from a donor, used as a treatment for severe liver diseases."
    },
    "['Antihistamines', 'Epinephrine', 'Corticosteroids', 'Antibiotics', 'Antifungal Cream']": {
        "Antihistamines": "Medications that relieve allergy symptoms by blocking histamine receptors, reducing itching, sneezing, and runny nose.",
        "Epinephrine": "Emergency medication used to treat severe allergic reactions (anaphylaxis) by relaxing muscles in the airways and tightening blood vessels.",
        "Corticosteroids": "Anti-inflammatory medications that reduce swelling and suppress the immune response in various conditions.",
        "Antibiotics": "Medications used to treat bacterial infections by killing or inhibiting the growth of bacteria.",
        "Antifungal Cream": "Topical medication used to treat fungal infections on the skin by inhibiting fungal growth."
    },
    "['Antibiotics', 'Proton Pump Inhibitors (PPIs)', 'H2 Blockers', 'Antacids', 'Cytoprotective agents']": {
        "Antibiotics": "Medications used to treat bacterial infections by killing or inhibiting the growth of bacteria.",
        "Proton Pump Inhibitors (PPIs)": "Medications that reduce stomach acid production by blocking the enzyme in the stomach wall that produces acid.",
        "H2 Blockers": "Medications that reduce stomach acid by blocking histamine receptors in the stomach lining.",
        "Antacids": "Over-the-counter medications that neutralize stomach acid to relieve heartburn and indigestion.",
        "Cytoprotective agents": "Medications that protect the lining of the stomach and intestines from damage by increasing mucus production or reducing acid."
    },
    "['Antiretroviral drugs', 'Protease inhibitors', 'Integrase inhibitors', 'Entry inhibitors', 'Fusion inhibitors']": {
        "Antiretroviral drugs": "Medications used to treat HIV by inhibiting the replication of the virus in the body.",
        "Protease inhibitors": "Class of antiretroviral drugs that prevent the HIV virus from maturing by inhibiting the protease enzyme.",
        "Integrase inhibitors": "Medications that block the integrase enzyme, preventing HIV from integrating its genetic material into the host cell's DNA.",
        "Entry inhibitors": "Medications that prevent HIV from entering human cells by blocking the receptors on the cell surface.",
        "Fusion inhibitors": "Antiretroviral drugs that prevent the HIV virus from fusing with the host cell membrane, blocking entry into the cell."
    },
    "['Insulin', 'Metformin', 'Sulfonylureas', 'DPP-4 inhibitors', 'GLP-1 receptor agonists']": {
        "Insulin": "Hormone therapy used to regulate blood sugar levels in people with diabetes by promoting the uptake of glucose into cells.",
        "Metformin": "Oral medication for type 2 diabetes that reduces glucose production in the liver and improves insulin sensitivity.",
        "Sulfonylureas": "Class of oral medications that stimulate the pancreas to release more insulin, used in the treatment of type 2 diabetes.",
        "DPP-4 inhibitors": "Medications that increase insulin production and decrease glucagon secretion by inhibiting the DPP-4 enzyme.",
        "GLP-1 receptor agonists": "Medications that mimic the incretin hormone GLP-1, enhancing insulin secretion and inhibiting glucagon release."
    },
    "['Antibiotics', 'Antiemetic drugs', 'Antidiarrheal drugs', 'IV fluids', 'Probiotics']": {
        "Antibiotics": "Medications used to treat bacterial infections by killing or inhibiting the growth of bacteria.",
        "Antiemetic drugs": "Medications that prevent or alleviate nausea and vomiting by blocking specific receptors in the brain and gut.",
        "Antidiarrheal drugs": "Medications that relieve diarrhea by slowing down intestinal movement or absorbing excess fluids.",
        "IV fluids": "Intravenous fluids used to rehydrate and provide essential nutrients and electrolytes in patients with severe dehydration.",
        "Probiotics": "Live beneficial bacteria and yeasts that promote a healthy balance of gut microbiota and support digestive health."
    },
    "['Bronchodilators', 'Inhaled corticosteroids', 'Leukotriene modifiers', 'Mast cell stabilizers', 'Anticholinergics']": {
        "Bronchodilators": "Medications that relax the muscles around the airways, making it easier to breathe in conditions like asthma and COPD.",
        "Inhaled corticosteroids": "Medications that reduce inflammation in the airways, used to control and prevent asthma symptoms.",
        "Leukotriene modifiers": "Medications that block the action of leukotrienes, reducing inflammation and constriction of airways in asthma.",
        "Mast cell stabilizers": "Medications that prevent the release of histamine and other inflammatory substances from mast cells, used in allergic conditions.",
        "Anticholinergics": "Medications that relax and enlarge the airways by blocking acetylcholine receptors, used in the treatment of COPD and asthma."
    },
    "['Antihypertensive medications', 'Diuretics', 'Beta-blockers', 'ACE inhibitors', 'Calcium channel blockers']": {
        "Antihypertensive medications": "Medications used to lower high blood pressure, reducing the risk of heart disease and stroke.",
        "Diuretics": "Medications that help the kidneys remove excess fluid and salt from the body, lowering blood pressure.",
        "Beta-blockers": "Medications that reduce blood pressure and heart rate by blocking the effects of adrenaline on the heart.",
        "ACE inhibitors": "Medications that lower blood pressure by relaxing blood vessels and reducing the production of angiotensin II.",
        "Calcium channel blockers": "Medications that relax blood vessels by preventing calcium from entering the cells of the heart and blood vessel walls."
    },
    "['Analgesics', 'Triptans', 'Ergotamine derivatives', 'Preventive medications', 'Biofeedback']": {
        "Analgesics": "Pain-relieving medications used to alleviate mild to moderate pain, including headaches and migraines.",
        "Triptans": "Medications specifically used to treat migraines by stimulating serotonin receptors, reducing inflammation and constricting blood vessels.",
        "Ergotamine derivatives": "Medications that treat migraines by constricting blood vessels in the brain, reducing blood flow and pain.",
        "Preventive medications": "Medications taken regularly to reduce the frequency and severity of migraines, including beta-blockers, antiepileptics, and antidepressants.",
        "Biofeedback": "A technique that helps individuals control physiological functions, such as muscle tension and heart rate, to reduce pain and stress."
    },
    "['Pain relievers', 'Muscle relaxants', 'Physical therapy', 'Neck braces', 'Corticosteroids']": {
        "Pain relievers": "Medications used to alleviate pain, including over-the-counter options like acetaminophen and ibuprofen.",
        "Muscle relaxants": "Medications that relieve muscle spasms and tightness by acting on the central nervous system or muscle fibers.",
        "Physical therapy": "Rehabilitation treatment involving exercises and techniques to improve mobility, strength, and function.",
        "Neck braces": "Devices worn around the neck to immobilize and support the cervical spine in cases of injury or surgery.",
        "Corticosteroids": "Anti-inflammatory medications that reduce swelling and suppress the immune response in various conditions."
    },
    "['Blood thinners', 'Clot-dissolving medications', 'Anticonvulsants', 'Physical therapy', 'Occupational therapy']": {
        "Blood thinners": "Medications that prevent blood clots from forming or growing, reducing the risk of stroke and other complications.",
        "Clot-dissolving medications": "Medications that dissolve existing blood clots, used in the treatment of conditions like stroke and pulmonary embolism.",
        "Anticonvulsants": "Medications used to prevent and control seizures by stabilizing electrical activity in the brain.",
        "Physical therapy": "Rehabilitation treatment involving exercises and techniques to improve mobility, strength, and function.",
        "Occupational therapy": "Therapeutic intervention aimed at helping individuals regain the ability to perform daily activities and improve their quality of life."
    },
    "['IV fluids', 'Blood transfusions', 'Liver transplant', 'Medications for itching', 'Antiviral medications']": {
        "IV fluids": "Intravenous fluids used to rehydrate and provide essential nutrients and electrolytes in patients with severe dehydration.",
        "Blood transfusions": "Procedure in which blood or blood products are transferred into a patient's bloodstream to replace lost components.",
        "Liver transplant": "Surgical procedure to replace a diseased liver with a healthy one from a donor, used as a treatment for severe liver diseases.",
        "Medications for itching": "Medications, including antihistamines and corticosteroids, used to relieve itching caused by various conditions.",
        "Antiviral medications": "Medications used to treat viral infections by inhibiting the replication of the virus in the body."
    },
    "['Antimalarial drugs', 'Antipyretics', 'Antiemetic drugs', 'IV fluids', 'Blood transfusions']": {
        "Antimalarial drugs": "Medications used to prevent and treat malaria by killing the malaria parasite in the blood.",
        "Antipyretics": "Medications that reduce fever by acting on the hypothalamus to lower body temperature.",
        "Antiemetic drugs": "Medications that prevent or alleviate nausea and vomiting by blocking specific receptors in the brain and gut.",
        "IV fluids": "Intravenous fluids used to rehydrate and provide essential nutrients and electrolytes in patients with severe dehydration.",
        "Blood transfusions": "Procedure in which blood or blood products are transferred into a patient's bloodstream to replace lost components."
    },
    "['Antiviral drugs', 'Pain relievers', 'IV fluids', 'Blood transfusions', 'Platelet transfusions']": {
        "Antiviral drugs": "Medications used to treat viral infections by inhibiting the replication of the virus in the body.",
        "Pain relievers": "Medications used to alleviate pain, including over-the-counter options like acetaminophen and ibuprofen.",
        "IV fluids": "Intravenous fluids used to rehydrate and provide essential nutrients and electrolytes in patients with severe dehydration.",
        "Blood transfusions": "Procedure in which blood or blood products are transferred into a patient's bloodstream to replace lost components.",
        "Platelet transfusions": "Procedure in which platelets are transfused into a patient's bloodstream to treat or prevent bleeding caused by low platelet counts."
    },
    "['Antibiotics', 'Antipyretics', 'Analgesics', 'IV fluids', 'Corticosteroids']": {
        "Antibiotics": "Medications used to treat bacterial infections by killing or inhibiting the growth of bacteria.",
        "Antipyretics": "Medications that reduce fever by acting on the hypothalamus to lower body temperature.",
        "Analgesics": "Pain-relieving medications used to alleviate mild to moderate pain, including headaches and migraines.",
        "IV fluids": "Intravenous fluids used to rehydrate and provide essential nutrients and electrolytes in patients with severe dehydration.",
        "Corticosteroids": "Anti-inflammatory medications that reduce swelling and suppress the immune response in various conditions."
    },
    "['Vaccination', 'Antiviral drugs', 'IV fluids', 'Blood transfusions', 'Liver transplant']": {
        "Vaccination": "Administration of a vaccine to stimulate the immune system to develop protection against specific infectious diseases.",
        "Antiviral drugs": "Medications used to treat viral infections by inhibiting the replication of the virus in the body.",
        "IV fluids": "Intravenous fluids used to rehydrate and provide essential nutrients and electrolytes in patients with severe dehydration.",
        "Blood transfusions": "Procedure in which blood or blood products are transferred into a patient's bloodstream to replace lost components.",
        "Liver transplant": "Surgical procedure to replace a diseased liver with a healthy one from a donor, used as a treatment for severe liver diseases."
    },
    "['Alcohol cessation', 'Corticosteroids', 'IV fluids', 'Liver transplant', 'Nutritional support']": {
        "Alcohol cessation": "Complete abstinence from alcohol consumption, used to treat and prevent liver disease and other alcohol-related conditions.",
        "Corticosteroids": "Anti-inflammatory medications that reduce swelling and suppress the immune response in various conditions.",
        "IV fluids": "Intravenous fluids used to rehydrate and provide essential nutrients and electrolytes in patients with severe dehydration.",
        "Liver transplant": "Surgical procedure to replace a diseased liver with a healthy one from a donor, used as a treatment for severe liver diseases.",
        "Nutritional support": "Provision of essential nutrients through diet, supplements, or medical nutrition therapy to improve health and recovery."
    },
    "['Antibiotics', 'Isoniazid', 'Rifampin', 'Ethambutol', 'Pyrazinamide']": {
        "Antibiotics": "Medications used to treat bacterial infections by killing or inhibiting the growth of bacteria.",
        "Isoniazid": "Antibiotic used to treat and prevent tuberculosis by inhibiting the synthesis of mycolic acids in the bacterial cell wall.",
        "Rifampin": "Antibiotic used to treat tuberculosis and other bacterial infections by inhibiting bacterial RNA synthesis.",
        "Ethambutol": "Antibiotic used to treat tuberculosis by inhibiting the synthesis of the bacterial cell wall.",
        "Pyrazinamide": "Antibiotic used in combination with other medications to treat tuberculosis by inhibiting the growth of Mycobacterium tuberculosis."
    },
    "['Antipyretics', 'Decongestants', 'Cough suppressants', 'Antihistamines', 'Pain relievers']": {
        "Antipyretics": "Medications that reduce fever by acting on the hypothalamus to lower body temperature.",
        "Decongestants": "Medications that relieve nasal congestion by shrinking swollen blood vessels in the nasal passages.",
        "Cough suppressants": "Medications that reduce coughing by acting on the cough center in the brain or the airways.",
        "Antihistamines": "Medications that relieve allergy symptoms by blocking histamine receptors, reducing itching, sneezing, and runny nose.",
        "Pain relievers": "Medications used to alleviate pain, including over-the-counter options like acetaminophen and ibuprofen."
    },
    "['Antibiotics', 'Antiviral drugs', 'Antifungal drugs', 'IV fluids', 'Oxygen therapy']": {
        "Antibiotics": "Medications used to treat bacterial infections by killing or inhibiting the growth of bacteria.",
        "Antiviral drugs": "Medications used to treat viral infections by inhibiting the replication of the virus in the body.",
        "Antifungal drugs": "Medications used to treat fungal infections by inhibiting the growth of fungi or killing fungal cells.",
        "IV fluids": "Intravenous fluids used to rehydrate and provide essential nutrients and electrolytes in patients with severe dehydration.",
        "Oxygen therapy": "Medical treatment that provides extra oxygen to patients with breathing difficulties or low oxygen levels in the blood."
    },
    "['Laxatives', 'Pain relievers', 'Warm baths', 'Cold compresses', 'High-fiber diet']": {
        "Laxatives": "Medications that promote bowel movements by stimulating the intestines or softening the stool.",
        "Pain relievers": "Medications used to alleviate pain, including over-the-counter options like acetaminophen and ibuprofen.",
        "Warm baths": "Therapeutic baths in warm water used to relax muscles, improve circulation, and relieve pain and stiffness.",
        "Cold compresses": "Therapeutic application of cold packs to reduce swelling, numb pain, and decrease inflammation.",
        "High-fiber diet": "Dietary approach that includes foods rich in fiber to promote healthy digestion and prevent constipation."
    },
    "['Nitroglycerin', 'Aspirin', 'Beta-blockers', 'Calcium channel blockers', 'Thrombolytic drugs']": {
        "Nitroglycerin": "Medication used to treat angina (chest pain) by relaxing and widening blood vessels, improving blood flow to the heart.",
        "Aspirin": "Medication used to reduce pain, fever, and inflammation, and to prevent blood clots by inhibiting platelet aggregation.",
        "Beta-blockers": "Medications that reduce heart rate and blood pressure by blocking the effects of adrenaline on the heart.",
        "Calcium channel blockers": "Medications that relax and widen blood vessels by inhibiting the influx of calcium ions into the muscle cells.",
        "Thrombolytic drugs": "Medications that dissolve blood clots, used in the treatment of conditions like stroke and heart attack."
    },
    "['Compression stockings', 'Exercise', 'Elevating the legs', 'Sclerotherapy', 'Laser treatments']": {
        "Compression stockings": "Elastic garments worn on the legs to improve blood flow and reduce swelling and discomfort in conditions like varicose veins.",
        "Exercise": "Physical activity that promotes overall health and improves circulation, muscle strength, and cardiovascular fitness.",
        "Elevating the legs": "Therapeutic practice of raising the legs to reduce swelling, improve blood flow, and relieve discomfort.",
        "Sclerotherapy": "Medical procedure that involves injecting a solution into veins to shrink and close them, used to treat varicose and spider veins.",
        "Laser treatments": "Medical procedures that use laser energy to treat various conditions, including skin issues, varicose veins, and eye problems."
    },
    "['Levothyroxine', 'Antithyroid medications', 'Beta-blockers', 'Radioactive iodine', 'Thyroid surgery']": {
        "Levothyroxine": "Synthetic thyroid hormone used to treat hypothyroidism by replacing or supplementing natural thyroid hormones.",
        "Antithyroid medications": "Medications used to treat hyperthyroidism by inhibiting the production of thyroid hormones.",
        "Beta-blockers": "Medications that reduce heart rate and blood pressure by blocking the effects of adrenaline on the heart.",
        "Radioactive iodine": "Treatment for hyperthyroidism that involves taking radioactive iodine to destroy overactive thyroid cells.",
        "Thyroid surgery": "Surgical procedure to remove part or all of the thyroid gland, used to treat conditions like thyroid cancer and hyperthyroidism."
    },
    "['Glucose tablets', 'Candy or juice', 'Glucagon injection', 'IV dextrose', 'Diazoxide']": {
        "Glucose tablets": "Fast-acting glucose sources used to treat hypoglycemia (low blood sugar) by quickly raising blood glucose levels.",
        "Candy or juice": "Simple carbohydrates that quickly raise blood sugar levels, used to treat hypoglycemia.",
        "Glucagon injection": "Emergency injection of the hormone glucagon, used to treat severe hypoglycemia by stimulating the release of glucose from the liver.",
        "IV dextrose": "Intravenous administration of glucose solution to rapidly raise blood sugar levels in cases of severe hypoglycemia.",
        "Diazoxide": "Medication used to treat hypoglycemia by inhibiting insulin release and stimulating glucose production."
    },
    "['Pain relievers', 'Exercise', 'Hot and cold packs', 'Joint protection', 'Physical therapy']": {
        "Pain relievers": "Medications used to alleviate pain, including over-the-counter options like acetaminophen and ibuprofen.",
        "Exercise": "Physical activity that promotes overall health and improves circulation, muscle strength, and cardiovascular fitness.",
        "Hot and cold packs": "Therapeutic application of heat or cold to relieve pain, reduce inflammation, and promote healing.",
        "Joint protection": "Techniques and devices used to reduce strain and prevent injury to the joints, especially in conditions like arthritis.",
        "Physical therapy": "Rehabilitation treatment involving exercises and techniques to improve mobility, strength, and function."
    },
    "['NSAIDs', 'Disease-modifying antirheumatic drugs (DMARDs)', 'Biologics', 'Corticosteroids', 'Joint replacement surgery']": {
        "NSAIDs": "Nonsteroidal anti-inflammatory drugs that reduce pain, inflammation, and fever by inhibiting the production of prostaglandins.",
        "Disease-modifying antirheumatic drugs (DMARDs)": "Medications that slow the progression of rheumatoid arthritis and other autoimmune diseases by modifying the immune response.",
        "Biologics": "Advanced medications derived from living cells that target specific components of the immune system to treat autoimmune diseases.",
        "Corticosteroids": "Anti-inflammatory medications that reduce swelling and suppress the immune response in various conditions.",
        "Joint replacement surgery": "Surgical procedure to replace a damaged joint with an artificial one, used to treat severe joint damage and pain."
    },
    "['Vestibular rehabilitation', 'Canalith repositioning', 'Medications for nausea', 'Surgery', 'Home exercises']": {
        "Vestibular rehabilitation": "Therapeutic program of exercises and activities designed to improve balance and reduce dizziness in patients with vestibular disorders.",
        "Canalith repositioning": "Series of head and body movements used to treat benign paroxysmal positional vertigo (BPPV) by repositioning displaced ear crystals.",
        "Medications for nausea": "Medications that prevent or alleviate nausea and vomiting by blocking specific receptors in the brain and gut.",
        "Surgery": "Medical procedure involving the cutting and repair of tissues, used to treat various conditions and injuries.",
        "Home exercises": "Physical activities and routines performed at home to improve strength, flexibility, and overall health."
    },
    "['Topical treatments', 'Antibiotics', 'Oral medications', 'Hormonal treatments', 'Isotretinoin']": {
        "Topical treatments": "Medications applied directly to the skin to treat conditions like acne, eczema, and psoriasis.",
        "Antibiotics": "Medications used to treat bacterial infections by killing or inhibiting the growth of bacteria.",
        "Oral medications": "Medications taken by mouth to treat various conditions, including infections, chronic diseases, and acute illnesses.",
        "Hormonal treatments": "Therapies that involve the use of hormones to treat conditions like acne, menopause symptoms, and hormone imbalances.",
        "Isotretinoin": "Powerful medication used to treat severe acne by reducing oil production and inflammation in the skin."
    },
    "['Antibiotics', 'Pain relievers', 'Antihistamines', 'Corticosteroids', 'Topical treatments']": {
        "Antibiotics": "Medications used to treat bacterial infections by killing or inhibiting the growth of bacteria.",
        "Pain relievers": "Medications used to alleviate pain, including over-the-counter options like acetaminophen and ibuprofen.",
        "Antihistamines": "Medications that relieve allergy symptoms by blocking histamine receptors, reducing itching, sneezing, and runny nose.",
        "Corticosteroids": "Anti-inflammatory medications that reduce swelling and suppress the immune response in various conditions.",
        "Topical treatments": "Medications applied directly to the skin to treat conditions like acne, eczema, and psoriasis."
    },
    "['Antibiotics', 'Urinary analgesics', 'Phenazopyridine', 'Antispasmodics', 'Probiotics']": {
        "Antibiotics": "Medications used to treat bacterial infections by killing or inhibiting the growth of bacteria.",
        "Urinary analgesics": "Medications that relieve pain, burning, and discomfort associated with urinary tract infections (UTIs).",
        "Phenazopyridine": "Medication used to relieve pain, burning, and discomfort caused by urinary tract infections (UTIs) and other urinary issues.",
        "Antispasmodics": "Medications that relieve muscle spasms in the gastrointestinal or urinary tract by relaxing smooth muscle tissue.",
        "Probiotics": "Live beneficial bacteria and yeasts that promote a healthy balance of gut flora and support digestive health."
    },
    "['Topical treatments', 'Phototherapy', 'Systemic medications', 'Biologics', 'Coal tar']": {
        "Topical treatments": "Medications applied directly to the skin to treat conditions like acne, eczema, and psoriasis.",
        "Phototherapy": "Medical treatment that uses ultraviolet light to treat skin conditions like psoriasis and eczema.",
        "Systemic medications": "Medications that affect the entire body, used to treat chronic diseases and severe conditions.",
        "Biologics": "Advanced medications derived from living cells that target specific components of the immune system to treat autoimmune diseases.",
        "Coal tar": "Topical treatment derived from coal used to treat skin conditions like psoriasis and eczema by reducing inflammation and scaling."
    },
    "['Topical antibiotics', 'Oral antibiotics', 'Antiseptics', 'Ointments', 'Warm compresses']": {
        "Topical antibiotics": "Antibiotics applied directly to the skin to treat bacterial infections like impetigo and minor wounds.",
        "Oral antibiotics": "Antibiotics taken by mouth to treat systemic bacterial infections.",
        "Antiseptics": "Substances that prevent the growth of microorganisms on the skin and other surfaces, used to clean wounds and prevent infection.",
        "Ointments": "Medicated creams or gels applied to the skin to treat various conditions and promote healing.",
        "Warm compresses": "Therapeutic application of heat to relieve pain, reduce swelling, and promote healing in injuries and infections."
    }
}

def getMedicines(dic, prediction):
    df = pd.read_csv(r'model\medications.csv')
    for key, value in dic.items():
        if value == prediction[0]:
            for i in range(len(df)):
                row = df.iloc[i, :2]
                if row[0] == key:
                    return row[1]
    return "No medicine found"

def suggestedDiets(dic,prediction):
    df=pd.read_csv(r'model\diets.csv')
    for key, value in dic.items():
        if value == prediction[0]:
            for i in range(len(df)):
                row = df.iloc[i, :2]
                if row[0] == key:
                    return row[1]
paragraph =recognize_speech()
print(paragraph)
print('\n')
if paragraph:
    found_symptoms = [symptom for symptom in symptoms if re.search(r'\b{}\b'.format(re.escape(symptom)), paragraph.lower())]
    symptom_vector = [1 if symptom in found_symptoms else 0 for symptom in symptoms]
    symptom_vector = np.array(symptom_vector).reshape(1, -1)
    prediction = model.predict(symptom_vector)
    disease_name = [key for key, value in disease_dict.items() if value == prediction[0]]
    
    if disease_name:
        print(f"You may have {disease_name[0]}")
        print('\n')
        key=disease_name[0]
        description = description_dict.get(key, 'Disease not found')
        print(description)
        print('\n')
        medicine = getMedicines(disease_dict, prediction)
        medicine_key=str(medicine)
        descriptions = medications_description.get(medicine_key, "Descriptions not found")
        ls=[]
        for med, desc in descriptions.items():
            print(f"{med}: {desc}")
#         print(f"Recommended treatment: {medicine}")
       
        print('\n')
        diets=suggestedDiets(disease_dict,prediction)
        print(f"Recommended diets: {diets}")
    else:                                                  
        print("Could not determine the disease.")