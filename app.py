from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model/scholarship_model.pkl')
df = pd.DataFrame()

# Sample scholarships list
scholarships = [
    {"id": 1, "name": "Merit-Based Scholarship", "description": "Awarded for academic excellence."},
    {"id": 2, "name": "Need-Based Scholarship", "description": "For students with financial needs."},
    {"id": 3, "name": "Extracurricular Excellence", "description": "For students with exceptional extracurricular activities."},
    {"id": 4, "name": "Sports Achievement Scholarship", "description": "For students excelling in sports."},
    {"id": 5, "name": "STEM Scholarship", "description": "For students pursuing science, technology, engineering, and mathematics fields."},
    {"id": 6, "name": "Women in Technology Scholarship", "description": "For female students pursuing a career in technology."},
    {"id": 7, "name": "Environmental Stewardship Scholarship", "description": "For students actively involved in environmental sustainability efforts."},
    {"id": 8, "name": "Leadership Scholarship", "description": "For students demonstrating outstanding leadership qualities."},
    {"id": 9, "name": "Community Service Scholarship", "description": "For students with a strong record of community service and volunteering."},
    {"id": 10, "name": "First-Generation College Student Scholarship", "description": "For first-generation college students in need of financial assistance."},
    {"id": 11, "name": "International Student Scholarship", "description": "For international students who demonstrate academic potential."},
    {"id": 12, "name": "Innovation and Entrepreneurship Scholarship", "description": "For students who show entrepreneurial spirit and innovation."},
    {"id": 13, "name": "Military Family Scholarship", "description": "For children of active-duty military personnel or veterans."},
    {"id": 14, "name": "Diversity and Inclusion Scholarship", "description": "For students who contribute to fostering diversity and inclusion on campus."},
    {"id": 15, "name": "Performing Arts Scholarship", "description": "For students excelling in music, dance, theater, or other performing arts."},
    {"id": 16, "name": "Research Excellence Scholarship", "description": "For students engaged in innovative research projects."},
    {"id": 17, "name": "Cultural Heritage Scholarship", "description": "For students who demonstrate a strong connection to their cultural heritage."},
    {"id": 18, "name": "Technology for Good Scholarship", "description": "For students who aim to use technology to solve social issues."},
    {"id": 19, "name": "Healthcare Career Scholarship", "description": "For students pursuing a career in healthcare or medicine."},
    {"id": 20, "name": "Arts and Humanities Scholarship", "description": "For students pursuing degrees in arts, literature, or history."}
]


@app.route('/')
def home():
    return render_template('home.html', scholarships=scholarships)

@app.route('/apply', methods=['GET', 'POST'])
def apply():
    def convert_caste(caste):
        if caste.lower() == 'oc':
            return 1  # OC (Open Category)
        elif caste.lower() == 'bc':
            return 2  # BC (Backward Class)
        elif caste.lower() in ['sc', 'st']:
            return 3  # SC or ST (Scheduled Caste or Scheduled Tribe)
        else:
            return 0  
    if request.method == 'POST':
        # Get user input
        details = {
            '10th_Percentage': float(request.form['10th_percentage']),
            '12th_Percentage': float(request.form['12th_percentage']),
            'BTech_Percentage': float(request.form['btech_percentage']),
            'Nationality': 1 if request.form['nationality'] == 'Indian' else 0,
            'Income': float(request.form['income']),
            'Physically_Disabled': 1 if request.form['physically_disabled'] == 'Yes' else 0,
            'Extracurricular': 1 if request.form['extracurricular'] == 'Yes' else 0,
            'Gender': 1 if request.form['gender'] == 'Male' else 0,
            # 'Caste': request.form['caste']
                        'Caste': convert_caste(request.form['caste'])
        }

        # Create a dataframe for prediction
        global df
        df = pd.DataFrame([details])
        
        # Predict eligibility
        prediction = model.predict(df)[0]
        eligibility = "Eligible" if prediction == 1 else "Not Eligible"

        # Redirect to results page
        return render_template('results.html', eligibility=eligibility, scholarships=scholarships)

    return render_template('apply.html')








@app.route('/scholarship/<int:id>')
def scholarship_details(id):
    scholarship = next((s for s in scholarships if s["id"] == id), None)
    
    if not scholarship:
        return redirect(url_for('home'))

    # Predict success chance (can be refined)
    global df
    prob = model.predict_proba(df)[0] 
    # success_chance = model.predict_proba(df)[0][1] * 100
    if len(prob) < 2:  # If only one class is predicted
        success_chance = 0  # Assign a default value if only class 0 is predicted
    else:
        success_chance = prob[1] * 100  

    return render_template('scholarship_detail.html', scholarship=scholarship, success_chance=success_chance)

if __name__ == '__main__':
    app.run(debug=True)
