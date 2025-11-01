import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
import matplotlib
import matplotlib.pyplot as plt
from resume_parser import parse_resume, get_job_description, calculate_similarity, find_non_matching_skills, calculate_ats_score
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Use Agg backend for Matplotlib
matplotlib.use('Agg')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-default-secret')

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Ramkr@123'
app.config['MYSQL_DB'] = 'Resume_Project'

mysql = MySQL(app)
bcrypt = Bcrypt(app)

# Global lists to store details
resume_details = []
similarity_scores = []
pie_chart_images = []
non_matching_skills_list = []

# Admin credentials (hashed for better security)
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = bcrypt.generate_password_hash("admin123").decode('utf-8')

def scrape_jobs(job_title, job_location):
    print(f"Scraping jobs for: {job_title} in {job_location}")

    if not job_title:
        job_title = ""
    if not job_location:
        job_location = ""

    job_title = job_title.replace(" ", "%20")  # URL encode spaces
    job_location = job_location.replace(" ", "%20")  # URL encode spaces
    url = f"https://www.linkedin.com/jobs/search?keywords={job_title}&location={job_location}"

    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)

    jobs = []
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'base-card')))
        job_cards = driver.find_elements(By.CLASS_NAME, 'base-card')[:10]

        for card in job_cards:
            try:
                title_element = card.find_element(By.CLASS_NAME, "base-search-card__title")
                company_element = card.find_element(By.CLASS_NAME, "base-search-card__subtitle")
                location_element = card.find_element(By.CLASS_NAME, "job-search-card__location")
                link_element = card.find_element(By.TAG_NAME, "a")

                title = title_element.text.strip() if title_element else "N/A"
                company = company_element.text.strip() if company_element else "N/A"
                location = location_element.text.strip() if location_element else "N/A"
                link = link_element.get_attribute("href") if link_element else "#"

                if title != "N/A" and company != "N/A":  # Avoid empty job entries
                    jobs.append({"title": title, "company": company, "location": location, "link": link})
                else:
                    print("Skipping incomplete job listing.")

            except Exception as e:
                print(f"Error extracting job data: {e}")

    except Exception as e:
        print(f"Error fetching job listings: {e}")

    driver.quit()
    return jobs
@app.route("/filter", methods=["GET", "POST"])
def home():
    jobs = []
    if request.method == "POST":
        job_title = request.form.get("job_title")
        job_location = request.form.get("job_location")  # Get location input
        jobs = scrape_jobs(job_title, job_location)
    
    return render_template("filter.html", jobs=jobs)

# Add admin credentials (can be stored in the database for production)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin123'  # You can hash this for better security


@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_loggedin'] = True
            flash('Admin login successful!', 'success')
            return redirect(url_for('dashboard'))  # Redirect to dashboard if credentials are correct
        else:
            flash('Invalid admin username or password', 'danger')

    return render_template('admin_login.html')

@app.route('/logout_admin')
def logout_admin():
    session.pop('admin_loggedin', None)
    flash('Admin logged out successfully', 'success')
    return redirect(url_for('admin_login'))

@app.route('/introduction')
def introduction():
    return render_template('introduction.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Check if the email already exists in the database
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            # If email exists, pass a flag to trigger the custom alert
            return render_template('signup.html', email_exists=True)

        # If the email does not exist, proceed with the signup
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, hashed_password))
        mysql.connection.commit()
        cursor.close()

        flash('Signup successful! You can now login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html', email_exists=False)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Retrieve user from the database
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()

        if user and bcrypt.check_password_hash(user[3], password):
            session['loggedin'] = True
            session['username'] = user[1]
            session['email'] = user[2]  # Store email in session
            session['user_id'] = user[0]  # Storing user ID in the session
            flash(f'Welcome {user[1]}!', 'success')
            return redirect(url_for('index'))  # Redirect to index after login
        else:
            flash('Login failed! Please check your email and password.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    session.pop('user_id', None)
    flash('You have successfully logged out.', 'success')
    return redirect(url_for('login'))

# Modify the index route in app.py to include ATS score
@app.route('/', methods=['GET', 'POST'])
def index():
    if 'loggedin' not in session:
        return redirect(url_for('introduction'))  # Redirect to introduction if not logged in

    global resume_details, similarity_scores, pie_chart_images, non_matching_skills_list

    selected_job_urls = []  # Store selected job URLs

    if request.method == 'POST':
        if 'search_jobs' in request.form:  # When the user searches for jobs
            job_title = request.form.get("job_title")
            job_location = request.form.get("job_location")

            # Get jobs based on the filter
            jobs = scrape_jobs(job_title, job_location)

            # Store only the top 5 job URLs
            selected_job_urls = [job["link"] for job in jobs[:1]]

            # Store in session for later use
            session["selected_job_urls"] = selected_job_urls

            flash("Top 5 job descriptions selected automatically!", "success")

        elif 'upload_resumes' in request.form:  # When the user uploads resumes
            resumes = request.files.getlist('resumes')

            # Clear previous results
            resume_details.clear()
            similarity_scores.clear()
            pie_chart_images.clear()
            non_matching_skills_list.clear()

            if "selected_job_urls" in session:
                selected_job_urls = session["selected_job_urls"]
            else:
                flash("No jobs selected. Please search for jobs first.", "danger")
                return redirect(url_for('index'))

            for job_url in selected_job_urls:
                job_description = get_job_description(job_url)

                for idx, resume in enumerate(resumes):
                    file_type = resume.filename.split('.')[-1].lower()
                    if file_type not in ['pdf', 'docx']:
                        continue  # Skip unsupported file types

                    details = parse_resume(resume, file_type)
                    resume_details.append(details)

                    # Calculate similarity
                    skills_score, education_score, beart_score,sbert_score,overall_score, precision, recall, f1 = calculate_similarity(details, job_description)
                    
                    # Calculate ATS score
                    ats_score = calculate_ats_score(precision, recall, f1)

                    similarity_scores.append({
        'skills_score': skills_score,
        'education_score': education_score,
        'beart_score': beart_score,  # Add BEART score
        'sbert_score': sbert_score,
        'overall_score': overall_score,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ATS Score': ats_score
    })

                    # Find non-matching skills
                    non_matching_skills = find_non_matching_skills(details['skills'], job_description)
                    non_matching_skills_list.append(non_matching_skills)

                    # Generate pie chart
                    labels = ['Similarity', 'Difference']
                    sizes = [overall_score, 100 - overall_score]
                    colors = ['#D5AAFF', '#4ECDC4']

                    plt.figure(figsize=(6, 6))
                    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
                    plt.axis('equal')

                    # Save the pie chart
                    chart_filename = f'similarity_pie_chart_{idx}.png'
                    chart_path = os.path.join('static', chart_filename)
                    plt.savefig(chart_path)
                    plt.close()

                    pie_chart_images.append(chart_filename)

            return redirect(url_for('candidate_details'))

    return render_template('index.html', selected_job_urls=selected_job_urls)

@app.route('/candidate_details')
def candidate_details():
    if not resume_details:
        return redirect(url_for('index'))

    return render_template('candidate_details.html', resume_details=resume_details, non_matching_skills=non_matching_skills_list)

@app.route('/dashboard')
def dashboard():
    if 'admin_loggedin' not in session:
        return redirect(url_for('admin_login'))  # Redirect to admin login if not logged in as admin

    # Retrieve all users from the database
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT id, username, email FROM users")
    candidates = cursor.fetchall()
    cursor.close()

    # Pass candidates to the template
    return render_template('dashboard.html', candidates=candidates)

@app.route('/similarity_score')
def similarity_score():
    if not resume_details or not similarity_scores:
        return redirect(url_for('index'))

    return render_template('similarity_score.html', resume_details=resume_details, similarity_scores=similarity_scores)

@app.route('/visualization_graph')
def visualization_graph():
    if not pie_chart_images:
        return redirect(url_for('index'))

    return render_template('visualization_graph.html', pie_chart_images=pie_chart_images)

@app.route('/similarity_bar_chart')
def similarity_bar_chart():
    if not similarity_scores:
        return redirect(url_for('index'))

    candidates = [f'Candidate {i+1}' for i in range(len(similarity_scores))]
    skills_scores = [score['skills_score'] for score in similarity_scores]
    education_scores = [score['education_score'] for score in similarity_scores]
    overall_scores = [score['overall_score'] for score in similarity_scores]

    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    index = range(len(candidates))

    plt.bar(index, skills_scores, width=bar_width, label='Skills', color='blue')
    plt.bar([i + bar_width for i in index], education_scores, width=bar_width, label='Education', color='green')
    plt.bar([i + 2 * bar_width for i in index], overall_scores, width=bar_width, label='Overall Similarity', color='purple')

    plt.xlabel('Candidates')
    plt.ylabel('Scores (%)')
    plt.title('Similarity Scores by Category')
    plt.xticks([i + bar_width for i in index], candidates)

    plt.legend()
    bar_chart_filename = 'similarity_bar_chart.png'
    bar_chart_path = os.path.join('static', bar_chart_filename)
    plt.savefig(bar_chart_path)
    plt.close()

    return render_template('similarity_bar_chart.html', bar_chart_image=bar_chart_filename)

if __name__ == '__main__':
    app.run(debug=True)