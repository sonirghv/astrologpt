import smtplib
import datetime
import csv
import os
import time
import pytz
import schedule
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import asyncio
import os

# Load environment variables
load_dotenv()

# ===== EMAIL CONFIGURATION =====
# Change these settings to match your email provider
SMTP_SERVER = "smtp.gmail.com" 
SMTP_PORT = 587 
SENDER_EMAIL = "healersmeetdev@gmail.com"  
SENDER_PASSWORD = "tgew tnlb flct davs"  
RECIPIENT_EMAIL = "support@healersmeet.com "  

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "HealersMeet"
COLLECTION_NAME = "new_users"

# Path for the CSV file
CSV_PATH = "new_clients.csv"

async def get_yesterdays_users():
    """Retrieve yesterday's new users from MongoDB."""
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # Get yesterday's date range in UTC
        today = datetime.datetime.now(pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today - datetime.timedelta(days=1)
        day_before = today - datetime.timedelta(days=2)
        
        # Query for users created yesterday (between day before yesterday and yesterday)
        cursor = collection.find({
            "last_updated": {
                "$gte": day_before.isoformat(),
                "$lt": today.isoformat()
            }
        })

        # Convert cursor to list
        users = await cursor.to_list(length=None)
        
        # Close MongoDB connection
        client.close()
        
        return users, yesterday.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Error retrieving users from MongoDB: {e}")
        yesterday_date = (datetime.datetime.now(pytz.UTC) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        return [], yesterday_date

def generate_csv_from_users(users, report_date):
    """Generate CSV file from user data."""
    # Update CSV path to include the report date
    csv_filename = f"new_clients_{report_date}.csv"
    
    # Prepare CSV data
    csv_data = [
        ["Name", "Email", "Mobile", "City", "Age", "Gender", "Join Date"]
    ]
    
    for user in users:
        csv_data.append([
            user.get("name", ""),
            user.get("email", ""),
            user.get("mobile", ""),
            user.get("city", ""),
            str(user.get("age", "")),
            user.get("gender", ""),
            report_date
        ])
    
    # Write to CSV file
    with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)
    
    return csv_filename

async def send_email():
    """Send email with CSV attachment."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Get yesterday's users
    users, report_date = await get_yesterdays_users()
    
    if not users:
        print(f"No new users found for {report_date}.")
        return False
    
    # Create message
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = f"New Clients Report for {report_date} - Generated on {today}"
    
    # Email body
    body = f"""Hello from Healers Meet,

Here is your report of new clients for {report_date}.

Total new clients: {len(users)}

The detailed information is attached in the CSV file.

Best regards,
Healers Meet Dev Team"""
    
    msg.attach(MIMEText(body, "plain"))
    
    # Generate and attach CSV file
    csv_file = generate_csv_from_users(users, report_date)
    
    try:
        with open(csv_file, "rb") as file:
            attachment = MIMEApplication(file.read(), Name=Path(csv_file).name)
        
        # Add header with filename
        attachment["Content-Disposition"] = f'attachment; filename="{Path(csv_file).name}"'
        msg.attach(attachment)
        
        # Connect to server and send email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"Email successfully sent to {RECIPIENT_EMAIL} with attachment")
        
        # Delete the CSV file after sending
        try:
            os.remove(csv_file)
            print(f"CSV file {csv_file} deleted successfully")
        except Exception as e:
            print(f"Warning: Failed to delete CSV file {csv_file}: {e}")
        
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        # Try to delete the CSV file even if email sending failed
        try:
            os.remove(csv_file)
            print(f"CSV file {csv_file} deleted after failed email attempt")
        except Exception as del_e:
            print(f"Warning: Failed to delete CSV file {csv_file}: {del_e}")
        return False

async def job():
    """Run the email sending job."""
    ist_now = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    yesterday_date = (datetime.datetime.now(pytz.UTC) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Running scheduled email job at {ist_now.strftime('%Y-%m-%d %H:%M:%S %Z')}... Sending report for {yesterday_date}")
    success = await send_email()
    if success:
        print("Email job completed successfully")
    else:
        print("Email job failed")

def convert_ist_to_local(ist_hour, ist_minute):
    """Convert 8:00 AM IST to local system time for scheduling."""
    # Get the IST time zone
    ist_tz = pytz.timezone('Asia/Kolkata')
    
    # Get the local system time zone
    local_tz = datetime.datetime.now().astimezone().tzinfo
    
    # Create a datetime object for today at 8:00 AM IST
    now = datetime.datetime.now()
    ist_time = ist_tz.localize(datetime.datetime(now.year, now.month, now.day, ist_hour, ist_minute))
    
    # Convert to local time
    local_time = ist_time.astimezone(local_tz)
    
    return local_time.strftime("%H:%M")

def run_scheduler():
    """Set up and run the scheduler for daily emails at 8:00 AM IST."""
    # Schedule the job to run daily at 8:00 AM IST (converted to local time for scheduling)
    local_time = convert_ist_to_local(8, 0)
    schedule.every().day.at(local_time).do(lambda: asyncio.run(job()))

    print(f"Email scheduler started. Will send emails daily at 8:00 AM IST (which is {local_time} in your local time).")
    print("Press Ctrl+C to exit.")

    # Run the job immediately once for testing (comment out if you don't want this)
    print("Running initial test job...")
    asyncio.run(job())

    # Keep the script running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\nScheduler stopped by user.")

async def test_email():
    """Run a single test email without scheduling."""
    print("Sending a test email...")
    success = await send_email()
    if success:
        print("Test email sent successfully!")
    else:
        print("Test email failed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Send daily emails with CSV attachments at 8:00 AM IST.")
    parser.add_argument("--test", action="store_true", help="Send a test email immediately without scheduling")
    
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(test_email())
    else:
        run_scheduler()