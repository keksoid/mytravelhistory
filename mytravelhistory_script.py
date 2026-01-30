import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from datetime import datetime
from google.auth.transport.requests import Request
import time

SCOPES = [
    'https://www.googleapis.com/auth/photoslibrary.readonly',
    'https://www.googleapis.com/auth/photoslibrary.readonly.appcreateddata',
    'https://www.googleapis.com/auth/photoslibrary.sharing',
    'https://www.googleapis.com/auth/photoslibrary.appendonly',
    'https://www.googleapis.com/auth/photoslibrary',
    'https://www.googleapis.com/auth/drive.readonly'
]

def main():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    # Build Drive service only (Photos API removed). We'll iterate all image files in Drive.
    drive_service = build('drive', 'v3', credentials=creds, static_discovery=False)

    print("Соединение установлено. Начинаю бесконечный цикл по файлам изображений в Google Drive...")

    def iter_all_drive_images(service, page_size=100):
        # Generator that yields Drive file dicts for image files (non-trashed).
        page_token = None
        q = "mimeType contains 'image/' and trashed=false"
        while True:
            resp = service.files().list(q=q, pageSize=page_size, fields='nextPageToken, files(id,name,mimeType,createdTime)').execute()
            files = resp.get('files', [])
            for f in files:
                yield f
            page_token = resp.get('nextPageToken')
            if not page_token:
                break

    try:
        cycle = 0
        while True:
            cycle += 1
            print(f"\n--- Starting cycle #{cycle} over Drive images ---")
            any_found = False
            for f in iter_all_drive_images(drive_service, page_size=100):
                any_found = True
                file_id = f.get('id')
                name = f.get('name')
                try:
                    meta = drive_service.files().get(fileId=file_id, fields='imageMediaMetadata, id, name, modifiedTime').execute()
                    created = meta.get('modifiedTime')
                    imm = meta.get('imageMediaMetadata', {})
                    loc = imm.get('location')
                    if loc:
                        lat = loc.get('latitude')
                        lon = loc.get('longitude')
                        print(f"{created} | {name} -> GPS: {lat}, {lon}")
                    else:
                        print(f"{created} | {name} -> no GPS in Drive metadata")
                except Exception as e:
                    print(f"Failed to fetch metadata for {name} ({file_id}): {e}")

            if not any_found:
                print("No image files found in Drive.")

            # Wait a short time before restarting the cycle to avoid tight loop and rate limits.
            time.sleep(5)

    except KeyboardInterrupt:
        print("\nStopped by user")

if __name__ == '__main__':
    main()