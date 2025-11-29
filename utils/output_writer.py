import os
from datetime import datetime

def write_html_report(content, title="UPSC News Digest Summary"):
    # Create output folder if not present
    os.makedirs("output", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/report_{timestamp}.html"

    html = f"""
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
            }}
            h1 {{
                color: #333;
            }}
            p {{
                white-space: pre-wrap;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p>{content}</p>
    </body>
    </html>
    """

    with open(filename, "w") as f:
        f.write(html)

    return filename
