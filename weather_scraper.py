"""
Weather Scraper Module - Fallback weather data collection
Scrapes weather data when APIs fail
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
from typing import Dict, Any, Optional


class WeatherScraper:
    """Scrape weather data from various sources as API fallback"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        self.timeout = 15
    
    def scrape_imd_weather(self, city: str) -> Dict[str, Any]:
        """Scrape weather from IMD (India Meteorological Department)"""
        try:
            # IMD city forecast page
            url = f"https://city.imd.gov.in/citywx/city_weather.php?id={self._get_imd_city_code(city)}"
            
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            if response.status_code != 200:
                return {'success': False, 'error': 'IMD not accessible'}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse weather data from IMD page
            weather_data = {}
            
            # Try to find temperature
            temp_elements = soup.find_all(string=re.compile(r'\d+\s*°C'))
            if temp_elements:
                temp_match = re.search(r'(\d+\.?\d*)\s*°C', str(temp_elements[0]))
                if temp_match:
                    weather_data['temperature'] = float(temp_match.group(1))
            
            # Try to find humidity
            humidity_elements = soup.find_all(string=re.compile(r'\d+\s*%'))
            if humidity_elements:
                humidity_match = re.search(r'(\d+)\s*%', str(humidity_elements[0]))
                if humidity_match:
                    weather_data['humidity'] = int(humidity_match.group(1))
            
            if weather_data:
                weather_data['success'] = True
                weather_data['source'] = 'imd.gov.in'
                weather_data['city'] = city
                return weather_data
            
            return {'success': False, 'error': 'Could not parse IMD data'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def scrape_timeanddate_weather(self, city: str, country: str = 'india') -> Dict[str, Any]:
        """Scrape weather from timeanddate.com"""
        try:
            # Format city name for URL
            city_slug = city.lower().replace(' ', '-')
            url = f"https://www.timeanddate.com/weather/{country}/{city_slug}"
            
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            if response.status_code != 200:
                return {'success': False, 'error': 'timeanddate.com not accessible'}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            weather_data = {
                'success': True,
                'source': 'timeanddate.com',
                'city': city
            }
            
            # Find temperature
            temp_div = soup.find('div', class_='h2')
            if temp_div:
                temp_text = temp_div.get_text()
                temp_match = re.search(r'(-?\d+)\s*°', temp_text)
                if temp_match:
                    weather_data['temperature'] = int(temp_match.group(1))
            
            # Find weather description
            desc_td = soup.find('td', id='qlook')
            if desc_td:
                desc_p = desc_td.find('p')
                if desc_p:
                    weather_data['description'] = desc_p.get_text().strip()
            
            # Find other details from table
            detail_table = soup.find('table', class_='table--left')
            if detail_table:
                rows = detail_table.find_all('tr')
                for row in rows:
                    th = row.find('th')
                    td = row.find('td')
                    if th and td:
                        label = th.get_text().strip().lower()
                        value = td.get_text().strip()
                        
                        if 'feels like' in label:
                            match = re.search(r'(-?\d+)', value)
                            if match:
                                weather_data['feels_like'] = int(match.group(1))
                        elif 'humidity' in label:
                            match = re.search(r'(\d+)\s*%', value)
                            if match:
                                weather_data['humidity'] = int(match.group(1))
                        elif 'wind' in label:
                            match = re.search(r'(\d+)\s*km/h', value)
                            if match:
                                weather_data['wind_speed'] = int(match.group(1))
                        elif 'visibility' in label:
                            match = re.search(r'(\d+)\s*km', value)
                            if match:
                                weather_data['visibility'] = int(match.group(1))
            
            if 'temperature' in weather_data:
                return weather_data
            
            return {'success': False, 'error': 'Could not parse weather data'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def scrape_weather_com(self, lat: float, lon: float) -> Dict[str, Any]:
        """Scrape weather from weather.com using coordinates"""
        try:
            url = f"https://weather.com/weather/today/l/{lat},{lon}"
            
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            if response.status_code != 200:
                return {'success': False, 'error': 'weather.com not accessible'}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            weather_data = {
                'success': True,
                'source': 'weather.com',
                'latitude': lat,
                'longitude': lon
            }
            
            # Find temperature (weather.com uses specific class names)
            temp_span = soup.find('span', {'data-testid': 'TemperatureValue'})
            if temp_span:
                temp_text = temp_span.get_text()
                temp_match = re.search(r'(-?\d+)', temp_text)
                if temp_match:
                    weather_data['temperature'] = int(temp_match.group(1))
            
            # Find weather phrase
            phrase_div = soup.find('div', {'data-testid': 'wxPhrase'})
            if phrase_div:
                weather_data['description'] = phrase_div.get_text().strip()
            
            # Find additional details
            details_section = soup.find('section', {'data-testid': 'TodaysDetailsSection'})
            if details_section:
                detail_items = details_section.find_all('div', {'data-testid': 'WeatherDetailsListItem'})
                for item in detail_items:
                    label_span = item.find('span', class_=re.compile('label'))
                    value_span = item.find('span', {'data-testid': 'TemperatureValue'}) or item.find('span', class_=re.compile('value'))
                    
                    if label_span and value_span:
                        label = label_span.get_text().strip().lower()
                        value = value_span.get_text().strip()
                        
                        if 'humidity' in label:
                            match = re.search(r'(\d+)', value)
                            if match:
                                weather_data['humidity'] = int(match.group(1))
                        elif 'wind' in label:
                            match = re.search(r'(\d+)', value)
                            if match:
                                weather_data['wind_speed'] = int(match.group(1))
                        elif 'uv' in label:
                            match = re.search(r'(\d+)', value)
                            if match:
                                weather_data['uv_index'] = int(match.group(1))
            
            if 'temperature' in weather_data:
                return weather_data
            
            return {'success': False, 'error': 'Could not parse weather.com data'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_weather(self, city: str = None, lat: float = None, lon: float = None) -> Dict[str, Any]:
        """
        Get weather using multiple scraping sources as fallback
        
        Args:
            city: City name
            lat: Latitude
            lon: Longitude
            
        Returns:
            Weather data dictionary
        """
        results = []
        
        # Try timeanddate.com first (most reliable for Indian cities)
        if city:
            result = self.scrape_timeanddate_weather(city)
            if result.get('success'):
                return self._format_weather_response(result)
            results.append(result)
        
        # Try weather.com with coordinates
        if lat and lon:
            result = self.scrape_weather_com(lat, lon)
            if result.get('success'):
                return self._format_weather_response(result)
            results.append(result)
        
        # Try IMD for Indian cities
        if city:
            result = self.scrape_imd_weather(city)
            if result.get('success'):
                return self._format_weather_response(result)
            results.append(result)
        
        # All sources failed
        return {
            'success': False,
            'error': 'All weather sources unavailable',
            'attempted_sources': [r.get('source', 'unknown') for r in results]
        }
    
    def _format_weather_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format scraped data to match API response format"""
        return {
            'current': {
                'temperature': data.get('temperature', 0),
                'feels_like': data.get('feels_like', data.get('temperature', 0)),
                'humidity': data.get('humidity', 0),
                'wind_speed': data.get('wind_speed', 0),
                'description': data.get('description', 'Weather data from web'),
                'city': data.get('city', 'Unknown'),
                'visibility': data.get('visibility', 10),
                'uv_index': data.get('uv_index', 0),
                'rainfall': data.get('rainfall', 0)
            },
            'source': data.get('source', 'web_scraper'),
            'timestamp': datetime.utcnow().isoformat(),
            'success': True
        }
    
    def _get_imd_city_code(self, city: str) -> str:
        """Get IMD city code for given city name"""
        # IMD city codes for major Indian cities
        city_codes = {
            'delhi': '42182',
            'new delhi': '42182',
            'mumbai': '43003',
            'chennai': '43279',
            'kolkata': '42809',
            'bangalore': '43296',
            'bengaluru': '43296',
            'hyderabad': '43150',
            'ahmedabad': '42647',
            'pune': '43063',
            'jaipur': '42348',
            'lucknow': '42369',
            'kanpur': '42452',
            'nagpur': '42867',
            'indore': '42667',
            'bhopal': '42667',
            'patna': '42492',
            'vadodara': '42634',
            'guwahati': '42410',
            'chandigarh': '42165',
            'coimbatore': '43311',
            'kochi': '43353',
            'visakhapatnam': '43189',
            'mysore': '43313',
            'mangalore': '43285',
            'hubli': '43198'
        }
        return city_codes.get(city.lower(), '42182')  # Default to Delhi


# Singleton instance
weather_scraper = WeatherScraper()
