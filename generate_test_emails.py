#!/usr/bin/env python
"""Generate diverse test emails for classification evaluation."""

import json
import base64
from datetime import datetime, timedelta
import random

def create_email(msg_id, subject, sender, body_html, days_ago=0):
    """Create a properly formatted email message."""
    date = (datetime.now() - timedelta(days=days_ago)).isoformat() + "Z"
    body_b64 = base64.b64encode(body_html.encode()).decode()
    snippet = body_html[:100].replace('<', '').replace('>', '')

    return {
        "id": msg_id,
        "subject": subject,
        "from": sender,
        "to": ["user@example.com"],
        "snippet": snippet,
        "body": body_b64,
        "date": date
    }

# Define test email categories
emails = []

# Category 1: Job/Career Related (10 emails)
job_emails = [
    create_email("job001", "Re: Senior Engineer position at Google",
                 "Recruiter <recruiter@google.com>",
                 "<p>Hi, We'd love to schedule a phone screen for the Senior Software Engineer role. Are you available next week?</p>", 1),
    create_email("job002", "Your LinkedIn profile was viewed by Microsoft",
                 "LinkedIn Jobs <jobs-noreply@linkedin.com>",
                 "<p>A recruiter from Microsoft viewed your profile. They're hiring for Cloud Infrastructure roles.</p>", 2),
    create_email("job003", "Congratulations on your offer!",
                 "HR Team <hr@stripe.com>",
                 "<p>We're excited to offer you the Staff Engineer position at Stripe. Please review the attached offer letter. Salary: $250,000. Start date: March 15.</p>", 3),
    create_email("job004", "Interview reminder: Technical round tomorrow",
                 "Amazon Recruiting <recruiting@amazon.com>",
                 "<p>This is a reminder about your technical interview tomorrow at 2 PM PT. Please prepare for system design and coding questions.</p>", 1),
    create_email("job005", "Thank you for interviewing with Anthropic",
                 "Talent <talent@anthropic.com>",
                 "<p>Thank you for taking the time to interview with us. We'll follow up within 3-5 business days with next steps.</p>", 5),
    create_email("job006", "Your application status update",
                 "Careers <careers@meta.com>",
                 "<p>Unfortunately, we've decided to move forward with other candidates for the ML Engineer role. We encourage you to apply for other positions.</p>", 7),
    create_email("job007", "Invitation to join our team",
                 "Sarah Chen <sarah@earlystart.io>",
                 "<p>Hey! I'm the CTO at EarlyStart, a seed-stage startup building AI tools. Would love to chat about our founding engineer opportunity.</p>", 2),
    create_email("job008", "LinkedIn: You appeared in 15 searches this week",
                 "LinkedIn <messages-noreply@linkedin.com>",
                 "<p>Recruiters are looking for candidates with your skills. Update your job preferences to get better matches.</p>", 3),
    create_email("job009", "Reminder: Complete your Workday onboarding",
                 "Workday <noreply@workday.com>",
                 "<p>Please complete your new hire paperwork before your start date. Missing items: I-9 verification, direct deposit setup.</p>", 1),
    create_email("job010", "Your Glassdoor review has been published",
                 "Glassdoor <noreply@glassdoor.com>",
                 "<p>Your review of TechCorp has been published. Thanks for helping the community make informed career decisions.</p>", 4),
]

# Category 2: Financial/Banking (8 emails)
finance_emails = [
    create_email("fin001", "Your Chase credit card statement is ready",
                 "Chase <alerts@chase.com>",
                 "<p>Your February statement is now available. Total balance: $2,547.32. Minimum payment due: $125 by March 5.</p>", 2),
    create_email("fin002", "Unusual activity on your account",
                 "Bank of America <alerts@bankofamerica.com>",
                 "<p>We noticed unusual activity on your checking account. A $500 charge at BestBuy.com. If this wasn't you, please call immediately.</p>", 1),
    create_email("fin003", "Your tax documents are ready",
                 "Robinhood <noreply@robinhood.com>",
                 "<p>Your 2024 1099-DIV and 1099-B forms are now available in your account. Total dividends: $432.18. Capital gains: $1,250.</p>", 15),
    create_email("fin004", "Rent payment successful",
                 "Zelle <noreply@zellepay.com>",
                 "<p>You sent $2,800 to John Smith for rent. Transaction ID: ZL7392847. Funds will arrive by end of business day.</p>", 5),
    create_email("fin005", "Your Venmo weekly summary",
                 "Venmo <venmo@venmo.com>",
                 "<p>This week you spent $143.50 and received $75. Top spending: Coffee ($48), Dinner ($65), Rideshare ($30.50).</p>", 7),
    create_email("fin006", "Important: Update your billing information",
                 "American Express <service@aexp.com>",
                 "<p>Your payment method for autopay was declined. Please update your bank account to avoid late fees.</p>", 1),
    create_email("fin007", "Your investment portfolio performance",
                 "Vanguard <client_services@vanguard.com>",
                 "<p>Your portfolio gained 2.3% this month. Total value: $145,280. Top performer: VTSAX (+3.1%).</p>", 3),
    create_email("fin008", "Fraud alert: Verify this transaction",
                 "Wells Fargo <alerts@wellsfargo.com>",
                 "<p>We blocked a $1,200 charge from an online casino. Reply YES to approve or NO to report fraud.</p>", 0),
]

# Category 3: Medical/Health (7 emails)
health_emails = [
    create_email("health001", "Your prescription is ready for pickup",
                 "CVS Pharmacy <pharmacy@cvs.com>",
                 "<p>Your prescription for Lisinopril 10mg is ready at CVS #4829. Copay: $15. Pickup by Feb 20.</p>", 1),
    create_email("health002", "Lab results available",
                 "Quest Diagnostics <results@questdiagnostics.com>",
                 "<p>Your recent blood work results are now available in the patient portal. All values appear normal. Review with your doctor.</p>", 2),
    create_email("health003", "Appointment confirmed: Dr. Martinez",
                 "ZocDoc <noreply@zocdoc.com>",
                 "<p>Your appointment with Dr. Elena Martinez (Primary Care) is confirmed for March 3 at 10:30 AM. Location: 123 Medical Plaza.</p>", 5),
    create_email("health004", "Your health insurance claim was processed",
                 "Blue Cross <claims@bluecross.com>",
                 "<p>Claim #BC-9482374 for office visit on Feb 5 has been processed. Insurance paid: $185. Your responsibility: $30 copay.</p>", 3),
    create_email("health005", "COVID-19 test results: Negative",
                 "Labcorp <noreply@labcorp.com>",
                 "<p>Your COVID-19 PCR test collected on Feb 12 is NEGATIVE. Keep this result for your records.</p>", 2),
    create_email("health006", "Reminder: Schedule your annual checkup",
                 "One Medical <reminders@onemedical.com>",
                 "<p>It's been over a year since your last physical. Schedule your annual wellness visit to stay on top of your health.</p>", 10),
    create_email("health007", "Your Fitbit weekly report",
                 "Fitbit <info@fitbit.com>",
                 "<p>This week you walked 52,431 steps (avg 7,490/day). Resting heart rate: 62 bpm. Sleep: 7h 15min average.</p>", 7),
]

# Category 4: Educational/Learning (6 emails)
education_emails = [
    create_email("edu001", "New course: Advanced Deep Learning",
                 "Coursera <noreply@coursera.org>",
                 "<p>Enroll in 'Advanced Deep Learning' by Andrew Ng. Learn transformers, GANs, and diffusion models. Starts March 10.</p>", 3),
    create_email("edu002", "Your assignment is due tomorrow",
                 "Canvas <notifications@instructure.com>",
                 "<p>Reminder: CS229 Machine Learning homework #3 is due tomorrow at 11:59 PM. Current submission status: Not submitted.</p>", 1),
    create_email("edu003", "Congratulations! You earned a certificate",
                 "Udacity <student@udacity.com>",
                 "<p>You've completed the Machine Learning Engineer Nanodegree! Download your certificate and share it on LinkedIn.</p>", 5),
    create_email("edu004", "Weekly digest: Papers you might like",
                 "arXiv <noreply@arxiv.org>",
                 "<p>New papers in cs.AI: 'Attention Is All You Need v2', 'Scaling Laws for Mixture of Experts', 'Constitutional AI Alignment'.</p>", 2),
    create_email("edu005", "Your O'Reilly learning subscription is expiring",
                 "O'Reilly Media <do-not-reply@oreilly.com>",
                 "<p>Your annual subscription expires in 7 days. Renew now to keep access to 60,000+ books and videos. Auto-renewal: OFF.</p>", 7),
    create_email("edu006", "New lecture uploaded: Lecture 15",
                 "Stanford Online <noreply@online.stanford.edu>",
                 "<p>CS231n Lecture 15 'Object Detection' is now available. Topics: R-CNN, Fast R-CNN, YOLO. Slides and code attached.</p>", 1),
]

# Category 5: Social/Personal (8 emails)
social_emails = [
    create_email("social001", "Mom wants to connect on Facebook",
                 "Facebook <notification@facebookmail.com>",
                 "<p>Susan Smith wants to be your friend on Facebook. You have 12 mutual friends.</p>", 2),
    create_email("social002", "You're invited: Sarah's Birthday Party",
                 "Eventbrite <noreply@eventbrite.com>",
                 "<p>Sarah Johnson invited you to her 30th birthday party on March 15 at 7 PM. Location: The Rooftop Bar. RSVP by March 10.</p>", 8),
    create_email("social003", "New match on Hinge!",
                 "Hinge <team@hinge.co>",
                 "<p>Alex liked your photo! Check out their profile and start a conversation.</p>", 1),
    create_email("social004", "Your Spotify Wrapped is here!",
                 "Spotify <no-reply@spotify.com>",
                 "<p>Your 2024 Spotify Wrapped is ready! You listened to 45,230 minutes of music. Top artist: Taylor Swift.</p>", 60),
    create_email("social005", "Dinner plans this weekend?",
                 "Jake <jake.wilson@gmail.com>",
                 "<p>Hey! Want to grab dinner this Saturday? There's a new ramen place in SoMa that's supposed to be amazing.</p>", 2),
    create_email("social006", "Your Strava weekly report",
                 "Strava <noreply@strava.com>",
                 "<p>This week: 3 runs, 15.2 miles, 2h 45min. You're ranked #23 in your city. New achievement: 10K in under 50 min!</p>", 7),
    create_email("social007", "New comment on your Instagram photo",
                 "Instagram <no-reply@instagram.com>",
                 "<p>@emily_travels commented on your photo: 'Beautiful sunset! Where is this?' Reply to keep the conversation going.</p>", 1),
    create_email("social008", "Your Goodreads 2024 reading challenge",
                 "Goodreads <noreply@goodreads.com>",
                 "<p>You've read 23 of 30 books this year! You're 77% to your goal. Currently reading: 'Project Hail Mary' by Andy Weir.</p>", 15),
]

# Category 6: E-commerce/Retail (10 emails)
ecommerce_emails = [
    create_email("shop001", "Your Warby Parker order has shipped",
                 "Warby Parker <hello@warbyparker.com>",
                 "<p>Great news! Your eyeglasses order #WP-5829 has shipped via UPS. Track: 1Z999AA10123456784. Delivery: Feb 18-20.</p>", 2),
    create_email("shop002", "20% off your next purchase",
                 "Everlane <team@everlane.com>",
                 "<p>Thanks for being a loyal customer! Enjoy 20% off your next order. Use code THANKS20 at checkout. Expires Feb 28.</p>", 5),
    create_email("shop003", "Your Costco receipt",
                 "Costco <receipts@costco.com>",
                 "<p>Thank you for shopping at Costco #428. Total: $287.43. Items: 12. Savings: $45.20. Scan your receipt for rewards.</p>", 1),
    create_email("shop004", "Price drop alert: MacBook Pro",
                 "Honey <alerts@joinhoney.com>",
                 "<p>The MacBook Pro 14' you're watching dropped $200 to $1,799 at B&H Photo. Lowest price in 90 days!</p>", 0),
    create_email("shop005", "Your Zappos return has been processed",
                 "Zappos <returns@zappos.com>",
                 "<p>We received your return of Nike Air Zoom Pegasus 40. Refund of $129.95 will appear in 3-5 business days.</p>", 3),
    create_email("shop006", "Back in stock: Patagonia fleece",
                 "Patagonia <email@patagonia.com>",
                 "<p>Good news! The Better Sweater Fleece in Navy (size M) is back in stock. Order soon—it's selling fast!</p>", 1),
    create_email("shop007", "Your Whole Foods delivery is on the way",
                 "Amazon Fresh <shipment@amazon.com>",
                 "<p>Your Whole Foods order is out for delivery. Arrival window: 6-8 PM today. 15 items, total: $94.32.</p>", 0),
    create_email("shop008", "Abandoned cart: Complete your purchase",
                 "IKEA <noreply@ikea.com>",
                 "<p>You left items in your cart! MALM bed frame ($299) and POÄNG chair ($149). Complete checkout in next 24h for free shipping.</p>", 1),
    create_email("shop009", "Your Nike order has been delivered",
                 "Nike <orders@nike.com>",
                 "<p>Your Nike order has been delivered to your front door. Order #NKE-4782931. Items: Air Force 1 Low (White, size 10).</p>", 1),
    create_email("shop010", "REI: Your dividend is ready",
                 "REI <dividend@rei.com>",
                 "<p>Your 2024 member dividend is $34.50! Use it on your next purchase or donate to outdoor nonprofits. Expires March 31.</p>", 30),
]

# Category 7: Subscriptions/Services (8 emails)
subscription_emails = [
    create_email("sub001", "Your Netflix payment failed",
                 "Netflix <info@netflix.com>",
                 "<p>We couldn't process your payment. Please update your payment method to avoid service interruption. Plan: Premium ($22.99/mo).</p>", 1),
    create_email("sub002", "New season of your favorite show!",
                 "Hulu <noreply@hulu.com>",
                 "<p>The Bear Season 3 is now streaming! Continue watching where you left off.</p>", 0),
    create_email("sub003", "Your Audible credit has arrived",
                 "Audible <noreply@audible.com>",
                 "<p>Your monthly credit is here! Browse 500,000+ titles. Recommendation: 'Atomic Habits' by James Clear (4.8 stars).</p>", 3),
    create_email("sub004", "Renewal reminder: NYTimes subscription",
                 "New York Times <noreply@nytimes.com>",
                 "<p>Your digital subscription renews in 7 days at $25/month. Manage your subscription or cancel anytime.</p>", 7),
    create_email("sub005", "Your Dropbox storage is almost full",
                 "Dropbox <no-reply@dropbox.com>",
                 "<p>You've used 98% of your 2TB storage. Upgrade to Dropbox Plus (3TB) for $11.99/month or delete some files.</p>", 2),
    create_email("sub006", "Adobe Creative Cloud renewal",
                 "Adobe <message@adobe.com>",
                 "<p>Your Creative Cloud subscription renews on Feb 28. Plan: All Apps ($59.99/mo). Update payment method if needed.</p>", 14),
    create_email("sub007", "Your Medium membership benefits",
                 "Medium <noreply@medium.com>",
                 "<p>As a member, you've unlocked unlimited reading this month. You read 23 stories and earned authors $12.40 in support.</p>", 5),
    create_email("sub008", "Welcome to GitHub Copilot!",
                 "GitHub <noreply@github.com>",
                 "<p>Your Copilot subscription is active! Start using AI pair programming in VS Code. Plan: Individual ($10/mo).</p>", 15),
]

# Combine all categories
all_emails = (job_emails + finance_emails + health_emails +
              education_emails + social_emails + ecommerce_emails +
              subscription_emails)

# Shuffle to mix categories
random.shuffle(all_emails)

# Write to JSONL file
output_file = "test-messages-new.jsonl"
with open(output_file, 'w') as f:
    for email in all_emails:
        f.write(json.dumps(email) + '\n')

print(f"✅ Generated {len(all_emails)} test emails")
print(f"📧 Saved to {output_file}")
print(f"\nCategory breakdown:")
print(f"  - Job/Career: {len(job_emails)}")
print(f"  - Financial: {len(finance_emails)}")
print(f"  - Medical/Health: {len(health_emails)}")
print(f"  - Education: {len(education_emails)}")
print(f"  - Social/Personal: {len(social_emails)}")
print(f"  - E-commerce: {len(ecommerce_emails)}")
print(f"  - Subscriptions: {len(subscription_emails)}")
print(f"  - Total: {len(all_emails)}")
