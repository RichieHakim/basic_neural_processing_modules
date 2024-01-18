
class Sender():
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