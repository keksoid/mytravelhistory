#!/usr/bin/env python3
"""
Local Photo Geolocation Aggregator
Scans a local folder for JPEG photos, extracts GPS coordinates,
reverse-geocodes them to countries, and outputs visited countries by date.
"""

import argparse
import sys
import logging
import time
import subprocess
import json
import math
import requests
from pathlib import Path
from datetime import datetime, timedelta

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def setup_cli_arguments():
    """
    Parse and validate CLI arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments with 'folder', 'years', and 'fast' attributes
    """
    parser = argparse.ArgumentParser(
        description="Extract visited countries from geotagged JPEG photos",
        epilog="Example: python process_local_photos.py --folder /path/to/photos --years 2 --fast"
    )
    
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to the folder containing JPEG photos (will search recursively)"
    )
    
    parser.add_argument(
        "--years",
        type=int,
        required=True,
        help="Number of years to look back from today"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Speed up processing by filtering files using file system dates instead of reading EXIF data. "
             "Assumes photos are organized by date taken (file modification time = photo capture time). "
             "Skips files outside the date range before EXIF extraction, resulting in ~2-3x faster processing."
    )
    
    args = parser.parse_args()
    return args


def validate_arguments(args):
    """
    Validate CLI arguments.
    
    Args:
        args (argparse.Namespace): Parsed CLI arguments
        
    Returns:
        tuple: (folder_path: Path, years: int, fast: bool) on success
        
    Raises:
        SystemExit: If validation fails
    """
    # Validate folder
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"Error: Folder '{args.folder}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not folder_path.is_dir():
        print(f"Error: '{args.folder}' is not a directory", file=sys.stderr)
        sys.exit(1)
    
    # Validate years
    if args.years <= 0:
        print(f"Error: Years must be a positive integer (got {args.years})", file=sys.stderr)
        sys.exit(1)
    
    fast = getattr(args, 'fast', False)
    
    return folder_path, args.years, fast


def calculate_cutoff_date(years):
    """
    Calculate the cutoff date for filtering photos.
    
    Args:
        years (int): Number of years to look back from today
        
    Returns:
        datetime: The cutoff date (today - N years)
    """
    today = datetime.now()
    cutoff_date = today - timedelta(days=years * 365)
    return cutoff_date


def iter_jpeg_files(folder_path, cutoff_date=None, apply_filter=False):
    """
    Generator that yields JPEG file paths from folder recursively,
    sorted by modification date in descending order (most recent first).
    Uses PowerShell for fast native Windows file system traversal.
    Optionally filters by file modification date on the PowerShell side.
    
    Args:
        folder_path (Path): Root folder to search
        cutoff_date (datetime): Optional cutoff date for filtering files
        apply_filter (bool): If True, only return files with modification time >= cutoff_date
        
    Yields:
        Path: Path to each JPEG file found, in descending date order
    """
    # Build the PowerShell command
    ps_command = f"""
    Get-ChildItem -Path '{folder_path}' -Filter '*.jpg' -Recurse -File -ErrorAction SilentlyContinue |
    """
    
    # Add date filtering if requested
    if apply_filter and cutoff_date:
        # Format cutoff date as ISO string for PowerShell datetime parsing
        cutoff_str = cutoff_date.strftime('%Y-%m-%dT%H:%M:%S')
        ps_command += f"""
    Where-Object {{ $_.LastWriteTime -ge [datetime]'{cutoff_str}' }} |
    """
    
    ps_command += """
    Sort-Object -Property LastWriteTime -Descending |
    Select-Object -ExpandProperty FullName
    """
    
    try:
        # Execute PowerShell command
        result = subprocess.run(
            ['powershell', '-NoProfile', '-Command', ps_command],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            logging.warning(f"PowerShell error: {result.stderr}")
            return
        
        # Parse output - each line is a file path
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                yield Path(line.strip())
    
    except subprocess.TimeoutExpired:
        logging.error("PowerShell command timed out")
    except Exception as e:
        logging.error(f"Error executing PowerShell command: {e}")


def extract_gps_coordinates(image):
    """
    Extract GPS coordinates from JPEG image EXIF data.
    
    Args:
        image (PIL.Image): PIL Image object
        
    Returns:
        tuple: (latitude, longitude) or (None, None) if not found
    """
    try:
        # Try to get IFD data
        try:
            exif = image.getexif()
        except (AttributeError, KeyError):
            # Fallback to deprecated method if getexif not available
            exif = image._getexif()
        
        if not exif:
            return None, None
        
        # Try to get GPS IFD (tag 34853)
        try:
            gps_ifd = exif.get_ifd(34853)
        except (KeyError, AttributeError):
            # Alternative: access GPS data directly from main EXIF
            gps_ifd = exif.get(34853, None)
            if not gps_ifd:
                return None, None
        
        # Extract GPS tags
        gps_latitude = gps_ifd.get(2)  # GPSLatitude
        gps_longitude = gps_ifd.get(4)  # GPSLongitude
        gps_latitude_ref = gps_ifd.get(1, 'N')  # GPSLatitudeRef
        gps_longitude_ref = gps_ifd.get(3, 'E')  # GPSLongitudeRef
        
        if not (gps_latitude and gps_longitude):
            return None, None
        
        # Convert rational tuples to decimal degrees
        def convert_to_degrees(gps_data):
            """Convert GPS data tuples (degrees, minutes, seconds) to decimal degrees."""
            if not gps_data or len(gps_data) < 3:
                return None
            
            d = gps_data[0]
            m = gps_data[1]
            s = gps_data[2]
            
            # Handle both Fraction and tuple formats
            if hasattr(d, 'numerator'):
                d = d.numerator / d.denominator if d.denominator else 0
            if hasattr(m, 'numerator'):
                m = m.numerator / m.denominator if m.denominator else 0
            if hasattr(s, 'numerator'):
                s = s.numerator / s.denominator if s.denominator else 0
            
            return d + m / 60 + s / 3600
        
        lat = convert_to_degrees(gps_latitude)
        lon = convert_to_degrees(gps_longitude)
        
        if lat is None or lon is None:
            return None, None
        
        # Apply reference direction
        if gps_latitude_ref == 'S':
            lat = -lat
        if gps_longitude_ref == 'W':
            lon = -lon
        
        return lat, lon
    
    except Exception as e:
        logging.debug(f"Error extracting GPS coordinates: {e}")
        return None, None


def extract_exif_data(file_path, cutoff_date=None):
    """
    Extract date and GPS coordinates from JPEG EXIF data.
    Only extracts GPS coordinates if date is within cutoff_date (optimization).
    
    Args:
        file_path (Path): Path to JPEG file
        cutoff_date (datetime): Optional cutoff date for filtering. If provided, 
                                GPS extraction is skipped for photos outside the range.
        
    Returns:
        dict: Dictionary with keys:
            - 'filename': str
            - 'date': datetime or None
            - 'latitude': float or None
            - 'longitude': float or None
    """
    result = {
        'filename': file_path.name,
        'date': None,
        'latitude': None,
        'longitude': None
    }
    
    try:
        image = Image.open(file_path)
        
        # Try to get EXIF data
        try:
            exif = image.getexif()
        except (AttributeError, KeyError):
            # Fallback to deprecated method
            exif_raw = image._getexif()
            if not exif_raw:
                mod_time = file_path.stat().st_mtime
                result['date'] = datetime.fromtimestamp(mod_time)
                return result
            exif = exif_raw
        
        if not exif:
            # Fallback to file modification time
            mod_time = file_path.stat().st_mtime
            result['date'] = datetime.fromtimestamp(mod_time)
            return result
        
        # Parse EXIF data for date
        exif_dict = {}
        try:
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif_dict[tag_name] = value
        except (TypeError, AttributeError):
            # If iteration fails, try alternative approach
            exif_dict = dict(exif) if exif else {}
        
        # Extract date (try DateTimeOriginal first, fallback to DateTime)
        date_str = exif_dict.get('DateTimeOriginal') or exif_dict.get('DateTime')
        if date_str:
            try:
                result['date'] = datetime.strptime(str(date_str), '%Y:%m:%d %H:%M:%S')
            except (ValueError, TypeError):
                logging.debug(f"Could not parse date string: {date_str}")
                mod_time = file_path.stat().st_mtime
                result['date'] = datetime.fromtimestamp(mod_time)
        else:
            # Fallback to file modification time
            mod_time = file_path.stat().st_mtime
            result['date'] = datetime.fromtimestamp(mod_time)
        
        # Optimization: Only extract GPS coordinates if photo is within date range
        if cutoff_date is None or (result['date'] and result['date'] >= cutoff_date):
            result['latitude'], result['longitude'] = extract_gps_coordinates(image)
        
    except Exception as e:
        logging.debug(f"Error reading EXIF from {file_path.name}: {e}")
        # Fallback to file modification time
        try:
            mod_time = file_path.stat().st_mtime
            result['date'] = datetime.fromtimestamp(mod_time)
        except:
            pass
    
    return result


def load_geocode_cache(cache_file='geocode_cache.json'):
    """
    Load geocoding cache from file.
    
    Args:
        cache_file (str): Path to cache JSON file
        
    Returns:
        dict: Cache dictionary mapping coordinates to countries
    """
    cache_path = Path(cache_file)
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load cache from {cache_file}: {e}")
    return {}


def save_geocode_cache(cache, cache_file='geocode_cache.json'):
    """
    Save geocoding cache to file.
    
    Args:
        cache (dict): Cache dictionary to save
        cache_file (str): Path to cache JSON file
    """
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Could not save cache to {cache_file}: {e}")


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two coordinates in kilometers using Haversine formula.
    
    Args:
        lat1, lon1 (float): First coordinate (degrees)
        lat2, lon2 (float): Second coordinate (degrees)
        
    Returns:
        float: Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def query_nominatim(lat, lon, retries=4):
    """
    Query Nominatim reverse geocoding API for location data.
    
    Args:
        lat, lon (float): Coordinates to geocode
        retries (int): Number of retries on transient errors
        
    Returns:
        dict: Nominatim response data or None if failed
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        'lat': lat,
        'lon': lon,
        'format': 'json',
        'zoom': 12,
        'accept-language': 'en'
    }
    headers = {
        'User-Agent': 'MyTravelHistory/1.0 (local photo geocoding)'
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                time.sleep(1)  # Rate limiting
                return response.json()
            elif response.status_code in [429, 500, 502, 503, 504]:
                # Transient error - retry with exponential backoff
                wait_time = 2 ** attempt
                logging.debug(f"Nominatim error {response.status_code}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.debug(f"Nominatim error {response.status_code}")
                return None
        
        except requests.exceptions.RequestException as e:
            logging.debug(f"Request error: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    
    return None


def extract_country(nominatim_response):
    """
    Extract country name from Nominatim reverse geocoding response.
    
    Args:
        nominatim_response (dict): Nominatim JSON response
        
    Returns:
        str: Country name or None if not found
    """
    if not nominatim_response:
        return None
    
    try:
        address = nominatim_response.get('address', {})
        
        # Primary: get country name
        country = address.get('country')
        if country:
            # If it's a multi-level address, extract just the country name (last element)
            if ',' in country:
                country = country.split(',')[-1].strip()
            return country
        
        # Fallback: try country_code
        country_code = address.get('country_code', '').upper()
        return country_code if country_code else None
    
    except Exception as e:
        logging.debug(f"Error extracting country: {e}")
        return None


def reverse_geocode_cached(lat, lon, cache, cache_threshold_km=1.0):
    """
    Reverse geocode coordinates to country with cache and proximity lookup.
    
    Args:
        lat, lon (float): Coordinates to geocode
        cache (dict): Geocoding cache
        cache_threshold_km (float): Distance threshold for reusing cached results
        
    Returns:
        tuple: (country, updated_cache)
    """
    coord_key = f"{lat:.6f},{lon:.6f}"
    
    # Check exact cache match
    if coord_key in cache:
        cached_result = cache[coord_key]
        return cached_result, cache
    
    # Check proximity cache - reuse nearby cached result
    for cached_coord, cached_country in cache.items():
        try:
            cached_lat, cached_lon = map(float, cached_coord.split(','))
            distance = haversine_km(lat, lon, cached_lat, cached_lon)
            
            if distance <= cache_threshold_km:
                # Reuse nearby cached result and add new entry
                cache[coord_key] = cached_country
                return cached_country, cache
        except ValueError:
            continue
    
    # Not in cache - query Nominatim
    logging.info(f"Querying Nominatim for {lat:.6f}, {lon:.6f}...")
    response = query_nominatim(lat, lon)
    
    if response:
        country = extract_country(response)
    else:
        country = None
    
    # Cache result (even if None, to avoid repeated queries)
    cache[coord_key] = country
    
    return country, cache


def filter_photos_by_date(photos, cutoff_date):
    """
    Filter photos to keep only those within the date range (after cutoff_date).
    
    Args:
        photos (list): List of dicts with 'date', 'latitude', 'longitude', 'filename' keys
        cutoff_date (datetime): Only include photos with date >= cutoff_date
        
    Returns:
        tuple: (valid_photos, filtered_out_count) where valid_photos are photos within date range
    """
    valid_photos = []
    filtered_out_count = 0
    
    for photo in photos:
        if photo['date'] and photo['date'] >= cutoff_date:
            valid_photos.append(photo)
        else:
            filtered_out_count += 1
    
    return valid_photos, filtered_out_count


def aggregate_countries_by_date(photos):
    """
    Aggregate visited countries with date ranges.
    Groups dates into periods, treating gaps without other country visits as continuous stays.
    
    Logic: If there's a gap between dates for the same country, but no photos from other 
    countries in that gap, the dates are considered part of the same trip.
    
    Args:
        photos (list): List of geocoded photos with 'country' and 'date' fields
        
    Returns:
        dict: {country_name: [(start_date, end_date), ...]} sorted by date per country
    """
    visits = {}
    
    # First collect all unique dates per country
    for photo in photos:
        country = photo.get('country')
        date = photo.get('date')
        
        if country and date:
            if country not in visits:
                visits[country] = set()
            visits[country].add(date.date())  # Store only the date part
    
    # Collect ALL dates across all countries to detect gaps
    all_dates = set()
    for dates_set in visits.values():
        all_dates.update(dates_set)
    
    # Now group dates into ranges, bridging gaps if no other country visited
    visits_with_ranges = {}
    for country, dates in visits.items():
        sorted_dates = sorted(dates)
        
        # Group dates into ranges
        ranges = []
        if sorted_dates:
            range_start = sorted_dates[0]
            range_end = sorted_dates[0]
            
            for current_date in sorted_dates[1:]:
                gap_days = (current_date - range_end).days
                
                if gap_days == 1:
                    # Consecutive days - extend range
                    range_end = current_date
                else:
                    # There's a gap - check if another country visited in the gap
                    gap_has_other_country = False
                    
                    # Check all dates in the gap
                    check_date = range_end + timedelta(days=1)
                    while check_date < current_date:
                        if check_date in all_dates and check_date not in dates:
                            # Another country has a photo on this date
                            gap_has_other_country = True
                            break
                        check_date += timedelta(days=1)
                    
                    if gap_has_other_country:
                        # Gap is bridged by another country - save current range and start new
                        ranges.append((range_start, range_end))
                        range_start = current_date
                        range_end = current_date
                    else:
                        # Gap is NOT bridged by other country - extend range across the gap
                        range_end = current_date
            
            # Add the last range
            ranges.append((range_start, range_end))
        
        visits_with_ranges[country] = ranges
    
    return visits_with_ranges


def format_output(visits):
    """
    Format aggregated visits with date ranges for display.
    
    Args:
        visits (dict): {country_name: [(start_date, end_date), ...]}
        
    Returns:
        list: Formatted output lines
    """
    output = []
    
    # Sort countries alphabetically
    for country in sorted(visits.keys()):
        ranges = visits[country]
        
        # Format date ranges
        range_strs = []
        for start_date, end_date in ranges:
            if start_date == end_date:
                # Single day visit
                range_strs.append(start_date.strftime('%Y-%m-%d'))
            else:
                # Multi-day visit
                range_strs.append(f"{start_date.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')}")
        
        # Create output line: Country: range1, range2, range3, ...
        line = f"{country}: {', '.join(range_strs)}"
        output.append(line)
    
    return output


def main():
    """Main entry point."""
    start_time = time.time()
    
    # Parse arguments
    args = setup_cli_arguments()
    
    # Validate arguments
    folder_path, years, fast = validate_arguments(args)
    
    # Calculate cutoff date
    cutoff_date = calculate_cutoff_date(years)
    
    # Load geocoding cache
    geocode_cache = load_geocode_cache()
    cache_size_before = len(geocode_cache)
    
    # Display configuration
    print(f"\n{'='*70}")
    print(f"Configuration:")
    print(f"  Folder: {folder_path.absolute()}")
    print(f"  Years: {years}")
    print(f"  Cutoff date: {cutoff_date.strftime('%Y-%m-%d')} (today - {years} years)")
    print(f"  Fast mode: {'ON (file system dates, read EXIF only for in-range files)' if fast else 'OFF (read EXIF date for all files)'}")
    print(f"  Geocoding cache: {cache_size_before} entries loaded")
    print(f"{'='*70}\n")
    
    # Phase 1: File discovery
    if fast:
        print(f"[1/5] Searching for JPEG files in date range '{folder_path.name}'...")
    else:
        print(f"[1/5] Searching for JPEG files in '{folder_path.name}'...")
    discovery_start = time.time()
    
    file_list = list(iter_jpeg_files(folder_path, cutoff_date=cutoff_date, apply_filter=fast))
    total_files_found = len(file_list)
    
    discovery_time = time.time() - discovery_start
    if fast:
        print(f"      [OK] Found {total_files_found} JPEG files (within date range) in {discovery_time:.2f} seconds\n")
    else:
        print(f"      [OK] Found {total_files_found} JPEG files (all) in {discovery_time:.2f} seconds\n")
    
    # Phase 2: EXIF extraction
    print(f"[2/5] Extracting EXIF data from {total_files_found} photos...")
    exif_start = time.time()
    
    photos = []
    for idx, photo_file in enumerate(file_list, 1):
        exif_data = extract_exif_data(photo_file, cutoff_date=cutoff_date)
        photos.append(exif_data)
        
        # Show progress every 10% or every 50 files
        if idx % max(total_files_found // 10, 50) == 0 or idx == total_files_found:
            progress_pct = (idx / total_files_found) * 100
            elapsed = time.time() - exif_start
            print(f"      Progress: {idx}/{total_files_found} ({progress_pct:.1f}%) - {elapsed:.1f}s elapsed")
    
    exif_time = time.time() - exif_start
    print(f"      [OK] EXIF extraction completed in {exif_time:.2f} seconds\n")
    
    # Phase 3: Date range filtering
    print(f"[3/5] Filtering photos by date range (cutoff: {cutoff_date.strftime('%Y-%m-%d')})...")
    filter_start = time.time()
    
    valid_photos, filtered_out = filter_photos_by_date(photos, cutoff_date)
    
    filter_time = time.time() - filter_start
    print(f"      [OK] Filtering completed in {filter_time:.2f} seconds\n")
    
    # Phase 4: Reverse geocoding
    print(f"[4/5] Reverse geocoding {len([p for p in valid_photos if p['latitude']])} photos to countries...")
    geocoding_start = time.time()
    
    geocoded_photos = []
    gps_photos = [p for p in valid_photos if p['latitude'] and p['longitude']]
    gps_count_total = len(gps_photos)
    
    for idx, photo in enumerate(gps_photos, 1):
        country, geocode_cache = reverse_geocode_cached(
            photo['latitude'], 
            photo['longitude'], 
            geocode_cache
        )
        photo['country'] = country
        geocoded_photos.append(photo)
        
        # Show progress
        if idx % max(gps_count_total // 10, 1) == 0 or idx == gps_count_total:
            progress_pct = (idx / gps_count_total) * 100 if gps_count_total > 0 else 0
            elapsed = time.time() - geocoding_start
            print(f"      Progress: {idx}/{gps_count_total} ({progress_pct:.1f}%) - {elapsed:.1f}s elapsed")
    
    geocoding_time = time.time() - geocoding_start
    print(f"      [OK] Geocoding completed in {geocoding_time:.2f} seconds\n")
    
    # Save updated cache
    save_geocode_cache(geocode_cache)
    cache_size_after = len(geocode_cache)
    
    # Phase 5: Aggregate and format results
    print(f"[5/5] Aggregating visits by country...")
    aggregate_start = time.time()
    
    visits = aggregate_countries_by_date(geocoded_photos)
    output_lines = format_output(visits)
    
    aggregate_time = time.time() - aggregate_start
    print(f"      [OK] Aggregation completed in {aggregate_time:.2f} seconds\n")
    
    # Summary statistics
    total_time = time.time() - start_time
    gps_count = sum(1 for p in valid_photos if p['latitude'] and p['longitude'])
    no_gps_count = len(valid_photos) - gps_count
    countries_found = len(visits)
    total_visit_periods = sum(len(ranges) for ranges in visits.values())
    
    # Display results
    print(f"{'='*70}")
    print(f"VISITED COUNTRIES:")
    print(f"{'='*70}")
    if output_lines:
        for line in output_lines:
            print(line)
    else:
        print("No countries found with geocoding data.")
    print(f"{'='*70}\n")
    
    print(f"{'='*70}")
    print(f"Processing Summary:")
    print(f"  Total photos found: {total_files_found}")
    if fast:
        print(f"  Note: Only files within date range were included (fast mode enabled)")
    print(f"  Photos outside date range: {filtered_out}")
    print(f"  Photos within date range: {len(valid_photos)}")
    print(f"    - With GPS coordinates: {gps_count}")
    print(f"    - Without GPS coordinates: {no_gps_count}")
    print(f"  Unique countries identified: {countries_found}")
    print(f"  Total visit periods (date ranges): {total_visit_periods}")
    print(f"\nGeocoding Cache:")
    print(f"  Before: {cache_size_before} entries")
    print(f"  After: {cache_size_after} entries")
    print(f"  New entries added: {cache_size_after - cache_size_before}")
    print(f"\nTiming:")
    print(f"  File discovery: {discovery_time:.2f} seconds")
    print(f"  EXIF processing: {exif_time:.2f} seconds")
    print(f"  Date filtering: {filter_time:.2f} seconds")
    print(f"  Reverse geocoding: {geocoding_time:.2f} seconds")
    print(f"  Aggregation: {aggregate_time:.2f} seconds")
    print(f"  Total time: {total_time:.2f} seconds")
    
    if gps_count > 0:
        avg_time_per_file = (exif_time / total_files_found) * 1000  # Convert to ms
        print(f"  Average time per file: {avg_time_per_file:.2f} ms")
    
    print(f"{'='*70}\n")
    print("Task 5-6 complete: Country aggregation and final output")


if __name__ == "__main__":
    main()
