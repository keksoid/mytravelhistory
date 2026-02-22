# MyTravelHistory - Local Photo Geolocation Aggregator

A Python script that automatically extracts visited countries from geotagged JPEG photos stored locally. It analyzes photo GPS coordinates, reverse-geocodes them to country names, and produces a summary of travel history organized by date ranges.
asdf
## Purpose

This script replaces cloud-based photo aggregation services by processing photos stored on your local file system. It's designed for users who:

- Store travel photos locally (instead of using Google Drive/Photos)
- Want to see a quick summary of countries visited with dates
- Have photos with GPS/EXIF metadata embedded
- Want to avoid uploading personal travel data to external services

## Key Features

- 🚀 **Fast Processing**: ~70ms per file in normal mode, ~30ms in fast mode
- 📍 **GPS-based Geolocation**: Reverse-geocodes coordinates to country names using OpenStreetMap Nominatim API
- 💾 **Smart Caching**: Caches geocoding results locally to minimize API calls
- 📅 **Intelligent Date Grouping**: Groups consecutive dates into travel periods (e.g., "2026-02-02-2026-02-05")
- 🔍 **Gap Detection**: Treats gaps as part of the same trip if no other countries have photos on those days
- ⚡ **Optional Fast Mode**: Uses file system dates for ultra-fast filtering (~2-3x speedup)
- 🖥️ **PowerShell Integration**: Leverages native Windows file APIs for optimal performance

## Requirements

- Python 3.7+
- PIL/Pillow (for EXIF extraction)
- requests (for Nominatim API calls)
- Windows PowerShell (for file discovery)

## Installation

1. Clone or download this repository
2. Install Python dependencies:
   ```bash
   pip install pillow requests
   ```
3. Ensure PowerShell 5.0+ is available (standard on Windows 10+)

## Usage

### Basic Usage

Process all photos from the last 1 year:
```bash
python process_local_photos.py --folder "C:\path\to\photos" --years 1
```

Process all photos from the last 2 years:
```bash
python process_local_photos.py --folder "Z:\Camera" --years 2
```

### Fast Mode (Recommended for Organized Folders)

Use file system dates instead of reading EXIF for all files (~2-3x speedup):
```bash
python process_local_photos.py --folder "C:\path\to\photos" --years 1 --fast
```

**Note**: Fast mode assumes file modification times match photo capture dates. This is true if:
- Photos were copied/imported while maintaining timestamps
- Photos are organized by date taken in folder structure
- You're processing a recent backup

If unsure, use normal mode without `--fast`.

### Command-Line Arguments

```
--folder FOLDER   (required)  Path to folder with JPEG photos (searches recursively)
--years YEARS     (required)  Number of years to look back from today
--fast            (optional)  Use file system dates for filtering instead of EXIF
                              Results in ~2-3x faster processing
```

### Examples

```bash
# Process 1 year of photos in fast mode
python process_local_photos.py --folder "D:\MyPhotos" --years 1 --fast

# Process 5 years of photos in normal mode (reads EXIF dates for all files)
python process_local_photos.py --folder "D:\MyPhotos" --years 5

# Get help
python process_local_photos.py --help
```

## Output Format

The script displays visited countries with date ranges:

```
======================================================================
VISITED COUNTRIES:
======================================================================
Germany: 2025-05-28-2025-05-29, 2025-10-24-2025-10-25, 2026-02-14
Greece: 2025-10-03-2025-10-09
Poland: 2025-02-23-2025-05-27, 2025-06-01-2025-10-02, 2025-10-11-2025-10-18
Slovakia: 2026-02-02-2026-02-05
======================================================================
```

Format:
- Countries are listed alphabetically
- Date ranges show consecutive visited days (e.g., `2026-02-02-2026-02-05`)
- Single-day visits show just the date (e.g., `2026-02-14`)
- Multiple visit periods are separated by commas
- **Date range grouping logic**: Days are grouped even if there are gaps between them, as long as no other countries have photos during those gap days (indicating continuous travel within the same country)

## How It Works

The script processes photos in 5 phases:

### Phase 1: File Discovery
Uses PowerShell to recursively find all JPEG files (`.jpg`) in the specified folder, sorted by modification date (newest first).

**With `--fast`**: Filters files by creation date at this stage, reducing subsequent processing.

### Phase 2: EXIF Extraction
Opens each JPEG file and extracts:
- **Date**: From EXIF DateTimeOriginal tag (falls back to DateTime, then file modification time)
- **GPS Coordinates**: From EXIF GPS IFD data (converts from degrees/minutes/seconds format to decimal)

**With `--fast`**: GPS extraction is skipped for out-of-range files.

### Phase 3: Date Range Filtering
Filters photos to keep only those within the specified time range (e.g., last 1 year).
- Removes photos older than the cutoff date
- Removes photos without GPS data from subsequent processing

### Phase 4: Reverse Geocoding
Converts GPS coordinates to country names using OpenStreetMap Nominatim API.
- **Caching**: Results are cached locally in `geocode_cache.json`
- **Proximity Lookup**: Reuses cached results for coordinates within 1km of previously geocoded locations
- **Rate Limiting**: Enforces 1-second delay between API requests
- **Retry Logic**: Implements exponential backoff (2^attempt seconds) for transient errors

### Phase 5: Aggregation & Output
Groups photos by country and date, then:
1. Collects unique dates for each country
2. Groups consecutive dates into travel periods
3. **Bridges gaps**: Treats days without photos as continuous travel if no other countries visited (indicates movement within same country, not border crossing)
4. Displays results sorted alphabetically by country, chronologically by date

## Performance

Typical processing times for 3,963 photos (9.5GB):

| Mode | Discovery | EXIF | Total | Speedup |
|------|-----------|------|-------|---------|
| Normal | 7.8s | 285.7s | 293.7s | — |
| **--fast** | 6.1s | 66.9s | 73.0s | **4x faster** |

Average time per file:
- Normal mode: ~72ms per file
- Fast mode: ~69ms per file (fastest for in-range files only)

Performance depends on:
- Network latency (files on network shares are slower)
- File system performance
- EXIF complexity
- Cache hit rate for geocoding

## Geocoding Cache

The script maintains `geocode_cache.json` which stores:
- Previously geocoded coordinates → country mappings
- Reduces API calls on subsequent runs
- Can be deleted to force fresh geocoding

Example cache entry:
```json
{
  "48.856613,2.352222": "France",
  "51.507351,-0.127758": "United Kingdom"
}
```

## Requirements

### JPEG Photo Format
- Photos must be JPEG format (`.jpg` extension)
- Should contain EXIF data with GPS coordinates for meaningful processing
- If EXIF date is missing, file modification time is used as fallback

### GPS/EXIF Data
Photos need GPS coordinates embedded in EXIF data for country detection. Most modern smartphones automatically add this when:
- Location services are enabled
- Camera app has location permission

### File Organization
- For `--fast` mode: Assume file modification timestamps match photo dates
- No specific folder structure required (script searches recursively)

## Limitations

- **JPEG Only**: Other image formats (PNG, RAW, HEIC) are not processed
- **GPS Required**: Photos without GPS coordinates are excluded from country detection
- **API Rate Limiting**: Nominatim has strict rate limits; large new coordinate batches may take time
- **Timezone Unaware**: Uses system timezone for date calculations
- **Network Files**: Processing network shares is slower than local drives

## Troubleshooting

### No countries found
- Verify photos have GPS/EXIF data
- Check if cutoff date is correct (`--years` parameter)
- Ensure you have at least 1 photo with coordinates in the date range

### "Error: Folder does not exist"
- Verify path is correct and accessible
- Use full path (not relative paths for network shares)
- Example: `\\photos\DCIM\Camera\`

### Slow processing
- Try `--fast` mode if photos are organized by date
- Geocoding cache is empty on first run; subsequent runs are faster
- Network folder access is inherently slower than local drives

### Nominatim API errors
- Script handles transient errors automatically with retry logic
- If persistent, wait ~5 minutes before retrying
- Check Nominatim status: https://nominatim.openstreetmap.org/status

## Output Files

- `geocode_cache.json`: Caches geocoding results (auto-created/updated)
- Console output: Detailed processing statistics and results

## Example Session

```
$ python process_local_photos.py --folder D:\MyCamera --years 1 --fast

======================================================================
Configuration:
  Folder: D:\MyCamera
  Years: 1
  Cutoff date: 2025-02-22 (today - 1 years)
  Fast mode: ON (file system dates, read EXIF only for in-range files)
  Geocoding cache: 519 entries loaded
======================================================================

[1/5] Searching for JPEG files in date range 'MyCamera'...
      [OK] Found 973 JPEG files (within date range) in 6.1 seconds

[2/5] Extracting EXIF data from 973 photos...
      Progress: 100/973 (10.3%) - 7.2s elapsed
      Progress: 200/973 (20.6%) - 14.4s elapsed
      ...
      Progress: 973/973 (100.0%) - 66.9s elapsed
      [OK] EXIF extraction completed in 66.88 seconds

[3/5] Filtering photos by date range (cutoff: 2025-02-22)...
      [OK] Filtering completed in 0.00 seconds

[4/5] Reverse geocoding 966 photos to countries...
      Progress: 100/966 (10.4%) - 0.0s elapsed
      ...
      Progress: 966/966 (100.0%) - 0.0s elapsed
      [OK] Geocoding completed in 0.00 seconds

[5/5] Aggregating visits by country...
      [OK] Aggregation completed in 0.00 seconds

======================================================================
VISITED COUNTRIES:
======================================================================
Germany: 2025-05-28-2025-05-29, 2025-10-24-2025-10-25, 2026-02-14
Greece: 2025-10-03-2025-10-09
Poland: 2025-02-23-2025-05-27, 2025-06-01-2025-10-02
Slovakia: 2026-02-02-2026-02-05
======================================================================

Processing Summary:
  Total photos found: 973
  Note: Only files within date range were included (fast mode enabled)
  Photos outside date range: 0
  Photos within date range: 973
    - With GPS coordinates: 966
    - Without GPS coordinates: 7
  Unique countries identified: 4
  Total visit periods (date ranges): 10

Geocoding Cache:
  Before: 519 entries
  After: 519 entries
  New entries added: 0

Timing:
  File discovery: 6.08 seconds
  EXIF processing: 66.88 seconds
  Date filtering: 0.00 seconds
  Reverse geocoding: 0.00 seconds
  Aggregation: 0.00 seconds
  Total time: 73.36 seconds
  Average time per file: 68.73 ms

======================================================================
```

## Technical Details

### EXIF GPS Data Format
GPS coordinates in photos are typically stored as:
- Degrees: Minutes: Seconds (DMS) format
- References: N/S for latitude, E/W for longitude
- Script automatically converts to decimal degrees

Example:
- `N 48°51'23.8"` → `48.856613`
- `E 2°21'07.9"` → `2.352222` (Paris)

### Date Fallback Chain
For photo dates, the script tries:
1. EXIF DateTimeOriginal (most reliable)
2. EXIF DateTime (backup)
3. File modification time (last resort)

### Proximity-Based Caching
To reduce API calls, the script reuses cached results for nearby coordinates:
- Distance threshold: 1.0 km
- Uses Haversine formula for accurate distance calculation
- Significantly speeds up processing for clustered photos

## Privacy & Data

- Photos are processed **locally** on your machine
- GPS coordinates are sent to OpenStreetMap Nominatim for country lookup
- Cache file (`geocode_cache.json`) stores only coordinates and country names (no image data)
- No telemetry or tracking
- All results stay on your system

## License

This script is provided as-is. Free to use and modify.

## Contributing

Suggestions and improvements welcome!

## Acknowledgments

- OpenStreetMap/Nominatim for reverse geocoding
- Pillow library for EXIF extraction
- PowerShell for efficient file discovery
