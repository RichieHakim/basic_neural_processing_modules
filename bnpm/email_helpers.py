
class Sender():
    def __init__(
        self,
        api_key=None,
        ):
        """
        Initialize the Sender object.
        RH 2022
        
        Args:
            api_key (str):
                The SendGrid API key.
                Get your API key at https://app.sendgrid.com/settings/api_keys.
        """
        import sendgrid

        self.sg = sendgrid.SendGridAPIClient(api_key=api_key)

    def send(
        self, 
        from_email='serverteeny@gmail.com', 
        to_emails=['serverteeny@gmail.com',], 
        subject='Subject', 
        content='Body',
        verbose=False,
        ):
        """
        Send an email.
        RH 2022

        Args:
            from_email (str):
                The email address of the sender.
            to_emails (list of str):
                The email addresses of the recipients.
            subject (str):
                The subject of the email.
            content (str):
                The content of the email.
            verbose (bool):
                Whether to print the response from SendGrid.

        Returns:
            The response from SendGrid.
        """
        from sendgrid.helpers.mail import To, From, Content, Mail
        self.from_email = From(from_email)
        self.to_emails = To(to_emails)
        self.subject = subject
        self.content = Content("text/plain", content)
        
        self.mail = Mail(self.from_email, self.to_emails, self.subject, self.content)
        
        self.response = self.sg.client.mail.send.post(request_body=self.mail.get())

        if verbose:
            print(self.response.status_code)
            print(self.response.body)
            print(self.response.headers)

        return self.response
