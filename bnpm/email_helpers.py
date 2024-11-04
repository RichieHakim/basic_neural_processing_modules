class Sender_sendgrid():
    def __init__(
        self,
        api_key=None,
        verbose=1,
    ):
        """
        Initialize the Sender object.
        RH 2022
        
        Args:
            api_key (str):
                The SendGrid API key.
                Get your API key at https://app.sendgrid.com/settings/api_keys.
            verbose (bool):
                Whether to print the response from SendGrid.
                False / 0: Do not print.
                True / 1: Print success or failure.
                2: Print success or failure and the response from SendGrid.
        """
        import sendgrid

        self.sg = sendgrid.SendGridAPIClient(api_key=api_key)
        self.verbose = verbose

    def send(
        self, 
        from_email='serverteeny@gmail.com', 
        to_emails=['serverteeny@gmail.com',], 
        subject='Subject', 
        content='Body',
        html_content=None,
        attachments=None,
        verbose=None,
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
            html_content (str):
                The HTML content of the email.
                If None, then the ``content`` is assumed to be plain text.
            attachments (list of tuple):
                Either a list of filepaths or a list of tuples containing the file content, file name, and file type.
                If a list of filepaths, then the file content, file name, and file type will be inferred.
                If a list of tuples, then:
                 Format: [(file_content, file_name, file_type), ].
                 Example of a PDF attachment: [(base64.b64encode(pdf_file.read()).decode(), 'file_name.pdf', 'application/pdf'), ]
                 Example of a PNG attachment: [(base64.b64encode(png_file.read()).decode(), 'file_name.png', 'image/png'), ]
            verbose (bool):
                Whether to print the response from SendGrid.
                If None, then must be defined in the __init__ method.
                False / 0: Do not print.
                True / 1: Print success or failure.
                2: Print success or failure and the response from SendGrid.

        Returns:
            The response from SendGrid.
        """
        from sendgrid.helpers.mail import To, From, Content, Mail, Attachment, FileContent, FileName, FileType, Disposition, ContentId
        
        message = Mail(
            from_email=From(from_email), 
            to_emails=To(to_emails), 
            subject=subject, 
            html_content=Content("text/plain", content) if html_content is None else Content("text/html", html_content)
        )

        if attachments is not None:
            if isinstance(attachments, list) is False:
                attachments = [attachments,]
            if isinstance(attachments[0], str):
                attachments = [_prepare_attachment(attachment) for attachment in attachments]
            elif isinstance(attachments[0], tuple):
                pass
            else:
                raise TypeError("Attachments must be a list of filepaths or a list of tuples containing the file content, file name, and file type.")
            for attachment in attachments:
                message.attachment = Attachment(
                    FileContent(attachment[0]), 
                    FileName(attachment[1]), 
                    FileType(attachment[2]), 
                    Disposition('attachment'), 
                    ContentId('Attachment')
                )

        response = self.sg.client.mail.send.post(request_body=message.get())

        if self.verbose is not None:
            verbose = self.verbose
        if verbose > 0:
            print(f"Succesfully sent email to {to_emails}.") if response.status_code == 202 else print(f"Failed to send email to {to_emails}.")
        if verbose > 1:
            print(response.status_code)
            print(response.body)
            print(response.headers)

        return response

def _prepare_attachment(file_path):
    """
    Prepare a file to be attached to an email.
    RH 2022

    Args:
        file_path (str):
            Path to the file to be attached.

    Returns:
        tuple:
            file_content (str):
                The file content.
            file_name (str):
                The file name.
            file_type (str):
                The file type.
    """
    import base64
    import mimetypes

    file_name = file_path.split('/')[-1]
    file_type = mimetypes.guess_type(file_path)[0]
    with open(file_path, 'rb') as f:
        file_content = base64.b64encode(f.read()).decode()

    return file_content, file_name, file_type


## SMTP approach
import smtplib
import ssl
import os
import mimetypes
from email.message import EmailMessage


class Sender:
    """
    A class to send emails using SMTP.
    RH 2024

    Args:
        smtp_server (str): 
            The SMTP server address.
        smtp_port (int): 
            The SMTP server port (587 for TLS, 465 for SSL).
        smtp_username (str): 
            Username for SMTP authentication.
        smtp_password (str): 
            Password for SMTP authentication.

    Notes:
        - To configure gmail settings to allow for SMTP, visit:
            https://myaccount.google.com/lesssecureapps to enable less secure
            apps and/or https://myaccount.google.com/apppasswords to get an app
            password.

    Example:
        ```python
        # Use the sender within a context manager to reuse the SMTP connection
        with Sender(
            smtp_server='smtp.gmail.com',
            smtp_port=587,
            smtp_username='your_email@gmail.com',
            smtp_password='your_app_password'
        ) as sender:
            response = sender.send(
                to_emails=['recipient@example.com'],
                subject='Test Email',
                content='This is a test email sent from Python.',
                attachments=['/path/to/attachment.pdf'],
                from_email='your_email@gmail.com',
                cc_emails=['cc_recipient@example.com'],
                bcc_emails=['bcc_recipient@example.com'],
                verbose=True
            )
            print(response)
        ```
    """
    def __init__(
        self, 
        smtp_server='smtp.example.com', 
        smtp_port=587, 
        smtp_username='user@example.com', 
        smtp_password='password'
    ):
        """
        Initializes the Sender object with SMTP server details.
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port  # Use 587 for TLS, 465 for SSL
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.server = None  # SMTP server connection

    def __enter__(self):
        """
        Establishes the SMTP connection and logs in when entering a context.
        """
        context = ssl.create_default_context()
        self.server = smtplib.SMTP(self.smtp_server, self.smtp_port)
        self.server.starttls(context=context)
        self.server.login(self.smtp_username, self.smtp_password)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Closes the SMTP connection when exiting a context.
        """
        if self.server:
            self.server.quit()
            self.server = None

    def _check_server_connection(self):
        """
        Checks if the server connection is established. If not, it establishes
        the connection and logs in.

        returns:
            bool: 
                True if the connection is established, False otherwise.
        """
        if self.server is not None:
            code, _ = self.server.noop()
            if code == 250:
                return True
            else:
                if self.verbose:
                    print('Server connection code not 250. Found:', code)
                return False
        else:
            return False

    def send(
        self, 
        to_emails, 
        subject, 
        content=None, 
        html_content=None, 
        attachments=None, 
        from_email=None,
        cc_emails=None,
        bcc_emails=None,
        headers=None,
        verbose=False
    ):
        """
        Sends an email with the given parameters.

        Args:
            to_emails (list or str): 
                Recipient's email address(es).
            subject (str): 
                Subject of the email.
            content (str, optional): 
                Plain text content of the email.
            html_content (str, optional): 
                HTML content of the email.
            attachments (list, optional): 
                List of file paths to attach.
            from_email (str, optional): 
                Sender's email address. If None, uses the SMTP username. The
                recipient server may override this if it does not match the
                authenticated user (to prevent spoofing).
            cc_emails (list or str, optional):
                CC recipient's email address(es).
            bcc_emails (list or str, optional):
                BCC recipient's email address(es).
            headers (dict, optional):
                Additional headers to add to the email message.
            verbose (bool, optional): 
                If True, prints status messages.

        Returns:
            dict: 
                A dictionary with 'status' and 'message' keys.
        """
        ## Input validation
        if content is None and html_content is None:
            raise ValueError("Provide at least one of 'content' or 'html_content'.")
        if not to_emails:
            raise ValueError("'to_emails' must be provided.")
        if isinstance(to_emails, str):
            to_emails = [to_emails]
        if cc_emails is None:
            cc_emails = []
        elif isinstance(cc_emails, str):
            cc_emails = [cc_emails]
        if bcc_emails is None:
            bcc_emails = []
        elif isinstance(bcc_emails, str):
            bcc_emails = [bcc_emails]
        if attachments is None:
            attachments = []

        ## Create the email message
        message = EmailMessage()
        message['From'] = from_email if from_email else self.smtp_username
        message['To'] = ', '.join(to_emails)
        message['Subject'] = subject

        ## Add CC and BCC
        if cc_emails:
            message['Cc'] = ', '.join(cc_emails)
        # Note: BCC recipients are not added to the message headers

        ## Add any additional headers
        if headers:
            for key, value in headers.items():
                message[key] = value

        ## Set email content
        if content and html_content:
            message.set_content(content)
            message.add_alternative(html_content, subtype='html')
        elif content:
            message.set_content(content)
        elif html_content:
            message.set_content('This is a fallback message in plain text.')
            message.add_alternative(html_content, subtype='html')

        ## Add attachments
        for file_path in attachments:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Attachment '{file_path}' not found.")
            ctype, encoding = mimetypes.guess_type(file_path)
            if (ctype is None) or (encoding is not None):
                ctype = 'application/octet-stream'
            maintype, subtype = ctype.split('/', 1)
            with open(file_path, 'rb') as f:
                file_data = f.read()
                file_name = os.path.basename(file_path)
                message.add_attachment(file_data, maintype=maintype, subtype=subtype, filename=file_name)

        ## Send the email
        try:
            # Check if server connection is established
            if self.server is None:
                # Establish connection (for backward compatibility)
                context = ssl.create_default_context()
                self.server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                self.server.starttls(context=context)
                self.server.login(self.smtp_username, self.smtp_password)
                close_connection = True
            else:
                close_connection = False

            all_recipients = to_emails + cc_emails + bcc_emails

            self.server.send_message(
                message,
                from_addr=from_email if from_email else self.smtp_username,
                to_addrs=all_recipients
            )

            if close_connection:
                self.server.quit()
                self.server = None

            if verbose:
                print('Email sent successfully.')
            return {'status': 'success', 'message': 'Email sent successfully.'}
        except Exception as e:
            if verbose:
                print(f'Failed to send email: {e}')
            return {'status': 'error', 'message': str(e)}
        
    def __call__(self, *args, **kwargs):
        """
        Calls the .send method with the given arguments. See .send docstring for
        details.
        """
        return self.send(*args, **kwargs)

    def __repr__(self):
        return f"Sender(smtp_server='{self.smtp_server}', smtp_port={self.smtp_port}, smtp_username='{self.smtp_username}', server status 250={self._check_server_connection()})"
