# Main ICU guidelines protocols:

def sepsis_protocol(patient_data):
    # Use SIRS criteria to identify patients with suspected sepsis
    sirs_score = 0
    if patient_data['temperature'] > 38.0 or patient_data['temperature'] < 36.0:
        sirs_score += 1
    if patient_data['heart_rate'] > 90:
        sirs_score += 1
    if patient_data['respiratory_rate'] > 20:
        sirs_score += 1
    if patient_data['white_blood_cells'] > 12000 or patient_data['white_blood_cells'] < 4000:
        sirs_score += 1
    if sirs_score >= 2:
        # Administer broad-spectrum antibiotics within 1 hour of recognition of sepsis
        # Monitor lactate levels and fluid balance
        # Consider vasopressors for hypotension
        return "Sepsis suspected"
    else:
        return "No sepsis suspected"

def shock_protocol(patient_data):
    # Use MAP and lactate levels to monitor for shock
    map = (2 * patient_data['diastolic_bp'] + patient_data['systolic_bp']) / 3
    if map < 60:
        # Administer fluids to achieve adequate perfusion pressure
        # Consider vasopressors for persistent hypotension
        # Monitor lactate levels and urine output
        return "Hypotension/shock suspected"
    else:
        return "No hypotension/shock suspected"

def ventilator_protocol(patient_data):
    # Use lung protective ventilation strategy
    tidal_volume = patient_data['weight'] * 6
    plateau_pressure = patient_data['peak_pressure'] - patient_data['peep']
    if tidal_volume > 8 or plateau_pressure > 30:
        # Adjust ventilator settings to achieve lower tidal volumes and plateau pressures
        # Consider prone positioning for patients with severe ARDS
        return "Lung protective ventilation strategy in use"
    else:
        return "Lung protective ventilation strategy not required"

def pain_management_protocol(patient_data):
    # Use a multimodal approach to pain management
    pain_score = patient_data['pain_score']
    if pain_score > 3:
        # Administer analgesics to achieve target pain score
        # Consider non-pharmacological interventions such as music therapy and relaxation techniques
        return "Pain management in progress"
    else:
        return "No pain management required"

def glycemic_control_protocol(patient_data):
    # Use insulin infusion to maintain blood glucose levels between 140-180 mg/dL
    glucose = patient_data['glucose']
    if glucose > 180:
        # Initiate insulin infusion and titrate to achieve target glucose range
        # Monitor for hypoglycemia and adjust insulin infusion rate as needed
        return "Insulin infusion for glycemic control in progress"
    else:
        return "No insulin infusion required for glycemic control"
