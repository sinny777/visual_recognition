# Python code to illustrate Sending mail 
# to multiple users 
# from your Gmail account 
import smtplib
  
# list of email_id to send the mail
li = ["sinny777@gmail.com"]
  
for dest in li:
    s = smtplib.SMTP('smtp.gmail.com', 587)
    # s = smtplib.SMTP('localhost')
    s.starttls()
    s.login("sinny777@gmail.com", "PASSWORD")
    message = "AI Model Training Complete"
    s.sendmail("sinny777@gmail.com", dest, message)
    s.quit()
