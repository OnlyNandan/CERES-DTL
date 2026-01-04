class CeresApp {
    constructor() {
        this.translations = {};
        this.currentLang = 'en';
        this.userData = window.userData || {};
        this.marketData = [];
        this.diaryEntries = [];
        this.init();
    }

    async init() {
        await this.loadTranslations();
        this.currentLang = this.userData.language || 'en';
        this.setLanguageSelector();
        this.updateUITranslations();
        this.setDefaultDiaryDate();
        
        if (document.getElementById('weather-content')) {
            this.loadWeather();
            this.loadMarketPrices();
            this.loadWeatherAlerts();
            this.loadGovernmentSchemes();
            this.loadAirQuality();
        }
    }

    setDefaultDiaryDate() {
        const dateInput = document.getElementById('diary-date');
        if (dateInput) {
            dateInput.value = new Date().toISOString().split('T')[0];
        }
    }

    async loadTranslations() {
        try {
            const response = await fetch('/api/translations');
            const data = await response.json();
            this.translations = data.translations;
            this.cropTranslations = data.crops;
        } catch (error) {
            console.error('Failed to load translations:', error);
        }
    }

    setLanguageSelector() {
        const selector = document.getElementById('language-selector');
        if (selector) {
            selector.value = this.currentLang;
        }
    }

    updateUITranslations() {
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            const translation = this.getTranslation(key);
            if (translation) {
                element.textContent = translation;
            }
        });
    }

    getTranslation(key) {
        if (this.translations[this.currentLang] && this.translations[this.currentLang][key]) {
            return this.translations[this.currentLang][key];
        }
        if (this.translations['en'] && this.translations['en'][key]) {
            return this.translations['en'][key];
        }
        return key;
    }

    getCropTranslation(crop) {
        const cropLower = crop.toLowerCase();
        if (this.cropTranslations && this.cropTranslations[cropLower]) {
            return this.cropTranslations[cropLower][this.currentLang] || 
                   this.cropTranslations[cropLower]['en'] || 
                   crop;
        }
        return crop;
    }

    async changeLanguage(lang) {
        this.currentLang = lang;
        this.updateUITranslations();
        
        try {
            await fetch('/api/user/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ language: lang })
            });
        } catch (error) {
            console.error('Failed to update language preference');
        }
        
        if (this.marketData.length > 0) {
            this.renderMarketPrices(this.marketData);
        }
        
        this.loadGovernmentSchemes();
    }

    async loadWeather() {
        const loadingEl = document.getElementById('weather-loading');
        const contentEl = document.getElementById('weather-content');
        const errorEl = document.getElementById('weather-error');
        const forecastSection = document.getElementById('forecast-section');

        loadingEl.classList.remove('hidden');
        contentEl.classList.add('hidden');
        errorEl.classList.add('hidden');
        forecastSection.classList.add('hidden');

        try {
            let url = '/api/weather?lang=' + this.currentLang;
            
            if (this.userData.latitude && this.userData.longitude) {
                url += `&lat=${this.userData.latitude}&lon=${this.userData.longitude}`;
            } else if (this.userData.city) {
                url += `&city=${encodeURIComponent(this.userData.city)}`;
            } else if (this.userData.district) {
                url += `&city=${encodeURIComponent(this.userData.district)}`;
            } else if (this.userData.state) {
                // Use state capital as fallback
                url += `&city=${encodeURIComponent(this.userData.state)}`;
            } else {
                // Default to Delhi if no location set
                url += `&lat=28.6139&lon=77.2090`;
            }

            const response = await fetch(url);
            const data = await response.json();

            if (data.error) {
                throw new Error(data.message);
            }

            this.renderWeather(data);
            loadingEl.classList.add('hidden');
            contentEl.classList.remove('hidden');
            contentEl.classList.add('fade-in');

            if (data.forecast && data.forecast.length > 0) {
                this.renderForecast(data.forecast);
                forecastSection.classList.remove('hidden');
                forecastSection.classList.add('fade-in');
            }

        } catch (error) {
            loadingEl.classList.add('hidden');
            errorEl.classList.remove('hidden');
        }
    }

    renderWeather(data) {
        const current = data.current;
        
        document.getElementById('temperature').textContent = current.temperature;
        document.getElementById('weather-description').textContent = current.description || '';
        document.getElementById('weather-city').textContent = current.city || '';
        document.getElementById('humidity').textContent = current.humidity + '%';
        document.getElementById('wind-speed').textContent = current.wind_speed + ' km/h';
        document.getElementById('feels-like').textContent = current.feels_like + '°C';
        document.getElementById('rainfall').textContent = (current.rainfall || 0) + ' mm';
        
        // UV Index
        const uvEl = document.getElementById('uv-index');
        if (uvEl && current.uv_index !== undefined) {
            uvEl.textContent = current.uv_index.toFixed(1);
        }
        
        // Sunrise/Sunset - handle both data structures
        if (data.daily) {
            if (data.daily.sunrise) document.getElementById('sunrise').textContent = data.daily.sunrise;
            if (data.daily.sunset) document.getElementById('sunset').textContent = data.daily.sunset;
        } else {
            if (current.sunrise) document.getElementById('sunrise').textContent = current.sunrise;
            if (current.sunset) document.getElementById('sunset').textContent = current.sunset;
        }
        
        // Weather icon - handle WeatherAPI.com icon URL or fallback to emoji
        const iconEl = document.getElementById('weather-icon');
        if (current.icon && current.icon.includes('http')) {
            iconEl.src = current.icon.replace('//', 'https://');
        } else {
            const weatherCode = current.weather_code || 0;
            iconEl.src = this.getWeatherIcon(weatherCode);
        }
        iconEl.alt = current.description || 'Weather';
        
        // Store weather data for other tools
        this.weatherData = data;
    }

    getWeatherIcon(code) {
        // WMO Weather Codes to icon mapping
        const iconMap = {
            0: '☀️', 1: '🌤️', 2: '⛅', 3: '☁️',
            45: '🌫️', 48: '🌫️',
            51: '🌧️', 53: '🌧️', 55: '🌧️',
            61: '🌧️', 63: '🌧️', 65: '🌧️',
            71: '🌨️', 73: '🌨️', 75: '🌨️',
            80: '🌧️', 81: '🌧️', 82: '🌧️',
            95: '⛈️', 96: '⛈️', 99: '⛈️'
        };
        
        // Create a data URL for the emoji
        const emoji = iconMap[code] || '🌤️';
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80">
            <text x="50%" y="50%" font-size="50" text-anchor="middle" dominant-baseline="central">${emoji}</text>
        </svg>`;
        return 'data:image/svg+xml,' + encodeURIComponent(svg);
    }

    renderForecast(forecast) {
        const container = document.getElementById('forecast-container');
        container.innerHTML = '';

        const days = [
            this.getTranslation('today'),
            this.getTranslation('tomorrow'),
            ...['', '', '', '', '']
        ];

        forecast.slice(0, 7).forEach((day, index) => {
            const date = new Date(day.date);
            const dayName = index < 2 ? days[index] : date.toLocaleDateString(this.currentLang === 'en' ? 'en-US' : 'hi-IN', { weekday: 'short' });
            
            const dayEl = document.createElement('div');
            dayEl.className = 'text-center p-2 bg-gray-50 rounded-xl min-w-[60px]';
            
            // Handle both WeatherAPI.com format and Open-Meteo format
            let iconHtml;
            if (day.icon && day.icon.includes('weatherapi')) {
                iconHtml = `<img src="${day.icon.replace('//', 'https://')}" class="w-10 h-10 mx-auto" alt="${day.description || 'Weather'}">`;
            } else {
                const weatherCode = day.weather_code || 0;
                const emoji = this.getWeatherEmoji(weatherCode);
                iconHtml = `<span class="text-2xl">${emoji}</span>`;
            }
            
            // Handle both temp formats
            const maxTemp = day.temp_max ?? day.max_temp ?? day.avg_temp ?? '--';
            const minTemp = day.temp_min ?? day.min_temp ?? '--';
            
            dayEl.innerHTML = `
                <p class="text-xs text-gray-500 mb-1">${dayName}</p>
                ${iconHtml}
                <p class="text-sm font-bold text-gray-900">${maxTemp}°</p>
                <p class="text-xs text-gray-500">${minTemp}°</p>
            `;
            container.appendChild(dayEl);
        });
    }

    getWeatherEmoji(code) {
        const emojiMap = {
            0: '☀️', 1: '🌤️', 2: '⛅', 3: '☁️',
            45: '🌫️', 48: '🌫️',
            51: '🌧️', 53: '🌧️', 55: '🌧️',
            61: '🌧️', 63: '🌧️', 65: '🌧️',
            71: '🌨️', 73: '🌨️', 75: '🌨️',
            80: '🌧️', 81: '🌧️', 82: '🌧️',
            95: '⛈️', 96: '⛈️', 99: '⛈️'
        };
        return emojiMap[code] || '🌤️';
    }

    async loadWeatherAlerts() {
        try {
            let url = '/api/weather/alerts?lang=' + this.currentLang;
            if (this.userData.latitude && this.userData.longitude) {
                url += `&lat=${this.userData.latitude}&lon=${this.userData.longitude}`;
            } else {
                // Default to Delhi
                url += `&lat=28.6139&lon=77.2090`;
            }

            const response = await fetch(url);
            const data = await response.json();

            if (data.success && data.alerts && data.alerts.length > 0) {
                this.renderAlerts(data.alerts);
            }
        } catch (error) {
            console.error('Failed to load weather alerts:', error);
        }
    }

    renderAlerts(alerts) {
        const container = document.getElementById('alerts-container');
        container.innerHTML = '';

        alerts.slice(0, 3).forEach(alert => {
            const alertEl = document.createElement('div');
            const alertClass = alert.severity === 'high' ? 'alert-high' : 
                             alert.severity === 'medium' ? 'alert-medium' : 'alert-low';
            
            const icons = {
                rain: '🌧️',
                heat: '🌡️',
                frost: '❄️',
                pest: '🐛',
                wind: '💨'
            };
            
            alertEl.className = `${alertClass} p-4 rounded-lg flex items-center gap-3 fade-in`;
            alertEl.innerHTML = `
                <span class="text-2xl">${icons[alert.type] || '⚠️'}</span>
                <div>
                    <p class="font-semibold text-gray-900">${alert.title}</p>
                    <p class="text-sm text-gray-600">${alert.message}</p>
                </div>
            `;
            container.appendChild(alertEl);
        });
    }

    async loadGovernmentSchemes() {
        try {
            const response = await fetch(`/api/gov-schemes?lang=${this.currentLang}`);
            const data = await response.json();

            if (data.success && data.schemes) {
                this.renderSchemes(data.schemes);
            }
        } catch (error) {
            console.error('Failed to load government schemes:', error);
        }
    }

    renderSchemes(schemes) {
        const container = document.getElementById('schemes-container');
        container.innerHTML = '';

        const colors = ['bg-green-50 border-green-200', 'bg-blue-50 border-blue-200', 
                       'bg-orange-50 border-orange-200', 'bg-purple-50 border-purple-200',
                       'bg-pink-50 border-pink-200', 'bg-yellow-50 border-yellow-200'];

        schemes.forEach((scheme, index) => {
            const schemeEl = document.createElement('div');
            schemeEl.className = `scheme-card p-4 rounded-xl border-2 ${colors[index % colors.length]} cursor-pointer`;
            schemeEl.innerHTML = `
                <h4 class="font-bold text-gray-900 mb-2">${scheme.name}</h4>
                <p class="text-sm text-gray-600 mb-3">${scheme.description}</p>
                <div class="flex items-center justify-between">
                    <span class="text-xs px-2 py-1 bg-white rounded-full text-gray-500">${scheme.type}</span>
                    ${scheme.link ? `<a href="${scheme.link}" target="_blank" class="text-green-600 text-sm font-medium hover:underline">Learn More →</a>` : ''}
                </div>
            `;
            container.appendChild(schemeEl);
        });
    }

    async loadAirQuality() {
        try {
            // First try to use weather data's air quality
            if (this.weatherData && this.weatherData.air_quality) {
                this.renderAirQuality(this.weatherData.air_quality);
                return;
            }
            
            // Fallback to dedicated endpoint
            let url = '/api/air-quality';
            if (this.userData.latitude && this.userData.longitude) {
                url += `?lat=${this.userData.latitude}&lon=${this.userData.longitude}`;
            } else {
                // Default to Delhi
                url += `?lat=28.6139&lon=77.2090`;
            }

            const response = await fetch(url);
            const data = await response.json();

            if (data.success) {
                this.renderAirQuality(data);
            }
        } catch (error) {
            console.error('Failed to load air quality:', error);
        }
    }

    renderAirQuality(data) {
        const aqiValue = document.getElementById('aqi-value');
        const aqiLabel = document.getElementById('aqi-label');
        const pm25 = document.getElementById('pm25');
        const pm10 = document.getElementById('pm10');

        // Handle both WeatherAPI format and Open-Meteo format
        const aqi = data.us_epa_index || data.aqi || (data.pm25 ? Math.round(data.pm25) : 0);
        aqiValue.textContent = Math.round(aqi);
        
        // AQI label based on US EPA index
        let label = 'Good';
        let color = 'text-green-600';
        if (aqi >= 6) { label = 'Hazardous'; color = 'text-purple-800'; }
        else if (aqi >= 5) { label = 'Very Unhealthy'; color = 'text-purple-600'; }
        else if (aqi >= 4) { label = 'Unhealthy'; color = 'text-red-600'; }
        else if (aqi >= 3) { label = 'Unhealthy (Sensitive)'; color = 'text-orange-600'; }
        else if (aqi >= 2) { label = 'Moderate'; color = 'text-yellow-600'; }
        
        aqiLabel.textContent = label;
        aqiValue.className = `text-4xl font-bold mb-2 ${color}`;
        
        // PM values
        const pm25Val = data.pm25 || data.pm2_5 || 0;
        const pm10Val = data.pm10 || 0;
        
        if (pm25Val) pm25.textContent = pm25Val.toFixed(1) + ' µg/m³';
        if (pm10Val) pm10.textContent = pm10Val.toFixed(1) + ' µg/m³';
    }

    async loadMarketPrices() {
        const loadingEl = document.getElementById('market-loading');
        const contentEl = document.getElementById('market-content');
        const errorEl = document.getElementById('market-error');

        loadingEl.classList.remove('hidden');
        contentEl.classList.add('hidden');
        errorEl.classList.add('hidden');

        try {
            let url = `/api/market-prices?state=${encodeURIComponent(this.userData.state || 'Karnataka')}&lang=${this.currentLang}`;
            
            if (this.userData.district) {
                url += `&district=${encodeURIComponent(this.userData.district)}`;
            }

            const response = await fetch(url);
            const data = await response.json();

            if (data.success && data.data) {
                this.marketData = data.data;
                this.renderMarketPrices(data.data);
                loadingEl.classList.add('hidden');
                contentEl.classList.remove('hidden');
                contentEl.classList.add('fade-in');
            } else {
                throw new Error('No data');
            }

        } catch (error) {
            loadingEl.classList.add('hidden');
            errorEl.classList.remove('hidden');
        }
    }

    renderMarketPrices(prices) {
        const tbody = document.getElementById('market-table-body');
        tbody.innerHTML = '';

        const filter = document.getElementById('market-commodity-filter').value;
        const filteredPrices = filter ? prices.filter(p => p.commodity === filter) : prices;

        filteredPrices.slice(0, 10).forEach(price => {
            const row = document.createElement('tr');
            row.className = 'table-row border-b border-gray-100 hover:bg-gray-50';
            
            const commodityName = this.getCropTranslation(price.commodity);
            
            row.innerHTML = `
                <td class="py-3 px-2">
                    <span class="font-semibold text-gray-900">${commodityName}</span>
                    <span class="text-xs text-gray-500 block">${price.variety || ''}</span>
                </td>
                <td class="py-3 px-2 text-gray-600 text-sm">${price.market}</td>
                <td class="py-3 px-2 text-right text-gray-600">₹${price.min_price.toLocaleString()}</td>
                <td class="py-3 px-2 text-right text-gray-600">₹${price.max_price.toLocaleString()}</td>
                <td class="py-3 px-2 text-right font-bold text-green-600">₹${price.modal_price.toLocaleString()}</td>
            `;
            tbody.appendChild(row);
        });

        if (filteredPrices.length === 0) {
            const row = document.createElement('tr');
            row.innerHTML = `<td colspan="5" class="py-8 text-center text-gray-500" data-i18n="no_data">${this.getTranslation('no_data')}</td>`;
            tbody.appendChild(row);
        }
    }

    filterMarketPrices() {
        if (this.marketData.length > 0) {
            this.renderMarketPrices(this.marketData);
        }
    }

    async getCropRecommendation(event) {
        event.preventDefault();

        const loadingEl = document.getElementById('recommendation-loading');
        const resultEl = document.getElementById('recommendation-result');
        const errorEl = document.getElementById('recommendation-error');

        loadingEl.classList.remove('hidden');
        resultEl.classList.add('hidden');
        errorEl.classList.add('hidden');

        const formData = {
            N: parseFloat(document.getElementById('nitrogen').value),
            P: parseFloat(document.getElementById('phosphorus').value),
            K: parseFloat(document.getElementById('potassium').value),
            ph: parseFloat(document.getElementById('ph').value),
            rainfall: parseFloat(document.getElementById('rainfall-input').value),
            lang: this.currentLang
        };

        try {
            const response = await fetch('/api/crop-recommendation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (data.error) {
                throw new Error(data.message);
            }

            this.renderRecommendation(data);
            loadingEl.classList.add('hidden');
            resultEl.classList.remove('hidden');
            resultEl.classList.add('fade-in');

        } catch (error) {
            loadingEl.classList.add('hidden');
            errorEl.classList.remove('hidden');
        }
    }

    renderRecommendation(data) {
        const cropName = data.recommended_crop_translated || this.getCropTranslation(data.recommended_crop);
        document.getElementById('recommended-crop').textContent = cropName;
        document.getElementById('confidence-value').textContent = data.confidence.toFixed(1) + '%';
        
        const confidenceBar = document.getElementById('confidence-bar');
        confidenceBar.style.width = '0%';
        setTimeout(() => {
            confidenceBar.style.width = data.confidence + '%';
        }, 100);

        // Load crop info
        this.loadCropInfo(data.recommended_crop);

        const topRecsContainer = document.getElementById('top-recommendations');
        topRecsContainer.innerHTML = '<p class="text-sm font-semibold text-gray-700 mb-2">Other Options:</p>';

        data.top_recommendations.slice(1, 3).forEach((rec, index) => {
            const recEl = document.createElement('div');
            recEl.className = 'flex items-center justify-between p-3 bg-gray-50 rounded-lg';
            recEl.innerHTML = `
                <span class="font-medium text-gray-700">${rec.crop_translated || this.getCropTranslation(rec.crop)}</span>
                <span class="text-sm text-gray-500">${rec.confidence.toFixed(1)}%</span>
            `;
            topRecsContainer.appendChild(recEl);
        });
    }

    async loadCropInfo(crop) {
        try {
            const response = await fetch(`/api/crop-info/${crop.toLowerCase()}?lang=${this.currentLang}`);
            const data = await response.json();

            if (data.success) {
                const panel = document.getElementById('crop-info-panel');
                const details = document.getElementById('crop-details');
                
                details.innerHTML = `
                    <div class="flex justify-between"><span class="text-gray-500">Season:</span><span class="font-medium">${data.season || 'N/A'}</span></div>
                    <div class="flex justify-between"><span class="text-gray-500">Water Needs:</span><span class="font-medium">${data.water_needs || 'N/A'}</span></div>
                    <div class="flex justify-between"><span class="text-gray-500">Duration:</span><span class="font-medium">${data.duration || 'N/A'}</span></div>
                `;
                panel.classList.remove('hidden');
            }
        } catch (error) {
            console.error('Failed to load crop info:', error);
        }
    }

    // Water Calculator
    async calculateWater(event) {
        event.preventDefault();
        
        const crop = document.getElementById('water-crop').value;
        const area = parseFloat(document.getElementById('water-area').value);

        try {
            const response = await fetch('/api/water-calculator', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ crop, area, lang: this.currentLang })
            });

            const data = await response.json();

            if (data.success) {
                const resultEl = document.getElementById('water-result');
                document.getElementById('water-amount').textContent = `${data.water_needed.toLocaleString()} liters/day`;
                document.getElementById('water-cycles').textContent = `Irrigation cycles: ${data.irrigation_cycles} per week`;
                resultEl.classList.remove('hidden');
            }
        } catch (error) {
            console.error('Failed to calculate water:', error);
        }
    }

    // Fertilizer Calculator
    async calculateFertilizer(event) {
        event.preventDefault();
        
        const crop = document.getElementById('fert-crop').value;
        const area = parseFloat(document.getElementById('fert-area').value);

        try {
            const response = await fetch('/api/fertilizer-calculator', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ crop, area, lang: this.currentLang })
            });

            const data = await response.json();

            if (data.success) {
                const resultEl = document.getElementById('fert-result');
                const detailsEl = document.getElementById('fert-details');
                
                detailsEl.innerHTML = `
                    <div class="flex justify-between p-2 bg-white rounded"><span>Urea</span><span class="font-bold">${data.urea} kg</span></div>
                    <div class="flex justify-between p-2 bg-white rounded"><span>DAP</span><span class="font-bold">${data.dap} kg</span></div>
                    <div class="flex justify-between p-2 bg-white rounded"><span>MOP</span><span class="font-bold">${data.mop} kg</span></div>
                    <p class="text-xs text-gray-500 mt-2">Based on ${data.npk.N}-${data.npk.P}-${data.npk.K} kg/ha requirement</p>
                `;
                resultEl.classList.remove('hidden');
            }
        } catch (error) {
            console.error('Failed to calculate fertilizer:', error);
        }
    }

    // Crop Calendar
    async loadCropCalendar() {
        try {
            const response = await fetch(`/api/crop-calendar?lang=${this.currentLang}`);
            const data = await response.json();

            if (data.success) {
                this.renderCropCalendar(data);
            }
        } catch (error) {
            console.error('Failed to load crop calendar:', error);
        }
    }

    renderCropCalendar(data) {
        const container = document.getElementById('calendar-content');
        container.innerHTML = '';

        const currentMonth = new Date().toLocaleString('en-US', { month: 'long' });
        
        // Current month info
        const monthEl = document.createElement('div');
        monthEl.className = 'p-4 bg-green-50 rounded-xl mb-4';
        monthEl.innerHTML = `
            <h4 class="font-bold text-green-800 mb-2">📅 ${currentMonth} - ${data.current_season}</h4>
            <p class="text-sm text-green-700">Best crops to sow now:</p>
        `;
        container.appendChild(monthEl);

        // Recommended crops for current season
        if (data.recommended_crops && data.recommended_crops.length > 0) {
            const cropsEl = document.createElement('div');
            cropsEl.className = 'grid grid-cols-2 gap-2 mb-4';
            data.recommended_crops.forEach(crop => {
                cropsEl.innerHTML += `
                    <div class="p-3 bg-white rounded-lg border text-center">
                        <p class="font-medium text-gray-900">${this.getCropTranslation(crop)}</p>
                    </div>
                `;
            });
            container.appendChild(cropsEl);
        }

        // Season-wise calendar
        const seasons = ['kharif', 'rabi', 'zaid'];
        const seasonNames = { kharif: 'Kharif (Jun-Oct)', rabi: 'Rabi (Nov-Mar)', zaid: 'Zaid (Mar-Jun)' };
        
        seasons.forEach(season => {
            if (data.calendar && data.calendar[season]) {
                const seasonEl = document.createElement('div');
                seasonEl.className = 'mb-3';
                seasonEl.innerHTML = `
                    <h5 class="font-semibold text-gray-700 mb-2">${seasonNames[season]}</h5>
                    <div class="flex flex-wrap gap-2">
                        ${data.calendar[season].map(crop => 
                            `<span class="px-3 py-1 bg-gray-100 rounded-full text-sm">${this.getCropTranslation(crop)}</span>`
                        ).join('')}
                    </div>
                `;
                container.appendChild(seasonEl);
            }
        });
    }

    // Farm Diary
    async loadFarmDiary() {
        try {
            const response = await fetch('/api/farm-diary');
            const data = await response.json();

            if (data.success) {
                this.diaryEntries = data.entries || [];
                this.renderDiaryEntries();
                this.updateDiarySummary(data.summary);
            }
        } catch (error) {
            console.error('Failed to load farm diary:', error);
        }
    }

    renderDiaryEntries() {
        const container = document.getElementById('diary-entries');
        container.innerHTML = '';

        if (this.diaryEntries.length === 0) {
            container.innerHTML = '<p class="text-center text-gray-500 py-4">No entries yet. Add your first entry above!</p>';
            return;
        }

        const activityIcons = {
            sowing: '🌱',
            irrigation: '💧',
            fertilizer: '🌿',
            pesticide: '🧪',
            harvest: '🌾',
            other: '📝'
        };

        this.diaryEntries.slice(0, 10).forEach(entry => {
            const entryEl = document.createElement('div');
            entryEl.className = 'p-3 bg-gray-50 rounded-lg flex items-start gap-3';
            entryEl.innerHTML = `
                <span class="text-xl">${activityIcons[entry.activity_type] || '📝'}</span>
                <div class="flex-1">
                    <div class="flex justify-between items-start">
                        <div>
                            <p class="font-medium text-gray-900">${entry.crop_name || 'General'}</p>
                            <p class="text-xs text-gray-500">${new Date(entry.date).toLocaleDateString()}</p>
                        </div>
                        <div class="text-right text-xs">
                            ${entry.expense ? `<p class="text-red-600">-₹${entry.expense}</p>` : ''}
                            ${entry.income ? `<p class="text-green-600">+₹${entry.income}</p>` : ''}
                        </div>
                    </div>
                    ${entry.notes ? `<p class="text-sm text-gray-600 mt-1">${entry.notes}</p>` : ''}
                </div>
            `;
            container.appendChild(entryEl);
        });
    }

    updateDiarySummary(summary) {
        if (!summary) return;
        
        document.getElementById('total-expense').textContent = `₹${(summary.total_expense || 0).toLocaleString()}`;
        document.getElementById('total-income').textContent = `₹${(summary.total_income || 0).toLocaleString()}`;
        
        const profit = (summary.total_income || 0) - (summary.total_expense || 0);
        const profitEl = document.getElementById('profit-loss');
        profitEl.textContent = `₹${Math.abs(profit).toLocaleString()}`;
        profitEl.className = `font-bold ${profit >= 0 ? 'text-green-600' : 'text-red-600'}`;
    }

    async addDiaryEntry(event) {
        event.preventDefault();

        const entry = {
            date: document.getElementById('diary-date').value,
            activity_type: document.getElementById('diary-activity').value,
            crop_name: document.getElementById('diary-crop').value,
            notes: document.getElementById('diary-notes').value,
            expense: parseFloat(document.getElementById('diary-expense').value) || 0,
            income: parseFloat(document.getElementById('diary-income').value) || 0
        };

        try {
            const response = await fetch('/api/farm-diary', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(entry)
            });

            const data = await response.json();

            if (data.success) {
                // Clear form
                document.getElementById('diary-crop').value = '';
                document.getElementById('diary-notes').value = '';
                document.getElementById('diary-expense').value = '';
                document.getElementById('diary-income').value = '';
                
                // Reload entries
                this.loadFarmDiary();
            }
        } catch (error) {
            console.error('Failed to add diary entry:', error);
        }
    }

    updateRangeValue(id) {
        const input = document.getElementById(id);
        const valueSpan = document.getElementById(id + '-value');
        if (input && valueSpan) {
            valueSpan.textContent = input.value;
        }
    }

    // === ADVANCED ML TOOLS ===
    
    // Disease Risk Analyzer
    async analyzeDiseaseRisk(event) {
        event.preventDefault();
        
        const data = {
            crop: document.getElementById('disease-crop').value,
            temperature: parseFloat(document.getElementById('disease-temp').value),
            humidity: parseFloat(document.getElementById('disease-humidity').value),
            rainfall: parseFloat(document.getElementById('disease-rainfall').value),
            consecutive_wet_days: parseInt(document.getElementById('disease-wetdays').value)
        };

        try {
            const response = await fetch('/api/ml/disease-risk', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                this.renderDiseaseRiskResult(result.data);
            } else {
                alert('Failed to analyze disease risk');
            }
        } catch (error) {
            console.error('Disease risk analysis failed:', error);
            alert('Error analyzing disease risk');
        }
    }

    renderDiseaseRiskResult(data) {
        const container = document.getElementById('disease-result');
        
        const riskColors = {
            low: { bg: 'bg-green-100', text: 'text-green-700', bar: 'bg-green-500' },
            moderate: { bg: 'bg-yellow-100', text: 'text-yellow-700', bar: 'bg-yellow-500' },
            high: { bg: 'bg-red-100', text: 'text-red-700', bar: 'bg-red-500' }
        };
        
        const colors = riskColors[data.risk_level] || riskColors.moderate;
        
        let html = `
            <div class="p-4 ${colors.bg} rounded-xl text-center">
                <p class="text-sm text-gray-600 mb-1">Overall Risk Level</p>
                <p class="text-3xl font-bold ${colors.text} mb-2">${data.risk_level.toUpperCase()}</p>
                <div class="w-full bg-gray-200 rounded-full h-3 mb-2">
                    <div class="${colors.bar} h-3 rounded-full transition-all duration-500" style="width: ${Math.min(data.overall_risk, 100)}%"></div>
                </div>
                <p class="text-sm ${colors.text}">Risk Score: ${data.overall_risk.toFixed(1)}%</p>
            </div>
            
            <div class="p-4 bg-gray-50 rounded-xl">
                <p class="font-semibold text-gray-700 mb-2">Primary Threat: <span class="text-red-600">${data.primary_threat?.replace(/_/g, ' ').toUpperCase() || 'None'}</span></p>
            </div>
        `;

        // Disease breakdown
        if (data.disease_risks && Object.keys(data.disease_risks).length > 0) {
            html += `<div class="space-y-2">`;
            for (const [disease, info] of Object.entries(data.disease_risks)) {
                const diseaseRiskPct = Math.min(info.risk_score, 100);
                html += `
                    <div class="p-3 bg-white border rounded-lg">
                        <div class="flex justify-between items-center mb-1">
                            <span class="font-medium text-gray-700">${info.disease.replace(/_/g, ' ')}</span>
                            <span class="text-sm font-bold ${diseaseRiskPct > 60 ? 'text-red-600' : diseaseRiskPct > 30 ? 'text-yellow-600' : 'text-green-600'}">${diseaseRiskPct.toFixed(0)}%</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div class="${diseaseRiskPct > 60 ? 'bg-red-500' : diseaseRiskPct > 30 ? 'bg-yellow-500' : 'bg-green-500'} h-2 rounded-full" style="width: ${diseaseRiskPct}%"></div>
                        </div>
                        ${info.contributing_factors ? `<p class="text-xs text-gray-500 mt-1">${info.contributing_factors.join(', ')}</p>` : ''}
                    </div>
                `;
            }
            html += `</div>`;
        }

        // Recommendations
        if (data.recommendations && data.recommendations.length > 0) {
            html += `
                <div class="p-4 bg-blue-50 rounded-xl">
                    <p class="font-semibold text-blue-800 mb-2">💡 Recommendations</p>
                    <ul class="text-sm text-blue-700 space-y-1">
                        ${data.recommendations.map(r => `<li>• ${r}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        // Spray Schedule
        if (data.spray_schedule) {
            html += `
                <div class="p-4 bg-orange-50 rounded-xl">
                    <p class="font-semibold text-orange-800 mb-2">🧪 Spray Schedule</p>
                    <p class="text-sm text-orange-700"><strong>Timing:</strong> ${data.spray_schedule.recommended_timing || 'N/A'}</p>
                    ${data.spray_schedule.avoid_spraying_if ? `<p class="text-xs text-orange-600 mt-1">⚠️ ${data.spray_schedule.avoid_spraying_if}</p>` : ''}
                </div>
            `;
        }

        container.innerHTML = html;
        container.classList.remove('hidden');
    }

    // Yield Predictor
    async predictYield(event) {
        event.preventDefault();
        
        const data = {
            crop: document.getElementById('yield-crop').value,
            state: this.userData.state || 'Karnataka',
            area_hectares: parseFloat(document.getElementById('yield-area').value),
            nitrogen: parseFloat(document.getElementById('yield-n').value),
            phosphorus: parseFloat(document.getElementById('yield-p').value),
            potassium: parseFloat(document.getElementById('yield-k').value),
            ph: parseFloat(document.getElementById('yield-ph').value),
            irrigation: document.getElementById('yield-irrigation').value,
            pest_pressure: document.getElementById('yield-pest').value,
            temperature: parseFloat(document.getElementById('yield-temp').value)
        };

        try {
            const response = await fetch('/api/ml/yield-prediction', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                this.renderYieldResult(result.data);
            } else {
                alert('Failed to predict yield');
            }
        } catch (error) {
            console.error('Yield prediction failed:', error);
            alert('Error predicting yield');
        }
    }

    renderYieldResult(data) {
        const container = document.getElementById('yield-result');
        
        const factors = data.factors || {};
        const limitingFactor = data.limiting_factor || 'unknown';
        
        let html = `
            <div class="p-4 bg-gradient-to-r from-amber-100 to-yellow-100 rounded-xl text-center">
                <p class="text-sm text-gray-600 mb-1">Predicted Yield</p>
                <p class="text-4xl font-bold text-amber-700">${data.predicted_yield?.toLocaleString() || 'N/A'}</p>
                <p class="text-sm text-gray-600">${data.unit || 'kg/ha'}</p>
                <p class="text-xs text-gray-500 mt-2">Range: ${data.yield_min?.toLocaleString()} - ${data.yield_max?.toLocaleString()} ${data.unit}</p>
            </div>
            
            <div class="p-4 bg-blue-50 rounded-xl">
                <p class="font-semibold text-blue-800 mb-2">📊 Confidence: ${(data.confidence || 0).toFixed(1)}%</p>
                <div class="w-full bg-blue-200 rounded-full h-3">
                    <div class="bg-blue-600 h-3 rounded-full" style="width: ${data.confidence || 0}%"></div>
                </div>
            </div>
        `;

        // Factor Analysis
        html += `
            <div class="p-4 bg-gray-50 rounded-xl">
                <p class="font-semibold text-gray-700 mb-3">🎯 Factor Analysis</p>
                <div class="space-y-2">
        `;
        
        const factorLabels = {
            water: '💧 Water',
            nutrients: '🌿 Nutrients',
            temperature: '🌡️ Temperature',
            soil_ph: '⚗️ Soil pH',
            pest_pressure: '🐛 Pest Impact'
        };
        
        for (const [factor, value] of Object.entries(factors)) {
            const pct = (value * 100).toFixed(0);
            const isLimiting = factor === limitingFactor;
            html += `
                <div class="flex items-center gap-2">
                    <span class="text-sm w-28 ${isLimiting ? 'text-red-600 font-bold' : 'text-gray-600'}">${factorLabels[factor] || factor} ${isLimiting ? '⚠️' : ''}</span>
                    <div class="flex-1 bg-gray-200 rounded-full h-2">
                        <div class="${pct >= 90 ? 'bg-green-500' : pct >= 70 ? 'bg-yellow-500' : 'bg-red-500'} h-2 rounded-full" style="width: ${pct}%"></div>
                    </div>
                    <span class="text-sm font-medium w-12 text-right">${pct}%</span>
                </div>
            `;
        }
        
        html += `</div></div>`;

        // Economic Projection
        if (data.economic_projection) {
            const econ = data.economic_projection;
            html += `
                <div class="p-4 bg-emerald-50 rounded-xl">
                    <p class="font-semibold text-emerald-800 mb-3">💰 Economic Projection</p>
                    <div class="grid grid-cols-2 gap-2 text-sm">
                        <div class="bg-white p-2 rounded-lg">
                            <p class="text-xs text-gray-500">Total Yield</p>
                            <p class="font-bold text-gray-800">${econ.total_yield_kg?.toLocaleString()} kg</p>
                        </div>
                        <div class="bg-white p-2 rounded-lg">
                            <p class="text-xs text-gray-500">Yield/Ha</p>
                            <p class="font-bold text-gray-800">${econ.yield_per_hectare_quintals?.toFixed(1)} qtl</p>
                        </div>
                        <div class="bg-white p-2 rounded-lg">
                            <p class="text-xs text-gray-500">Market Price</p>
                            <p class="font-bold text-gray-800">₹${econ.market_price_per_quintal?.toLocaleString()}/qtl</p>
                        </div>
                        <div class="bg-white p-2 rounded-lg">
                            <p class="text-xs text-gray-500">Est. Revenue</p>
                            <p class="font-bold text-emerald-600">₹${econ.estimated_revenue?.toLocaleString()}</p>
                        </div>
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;
        container.classList.remove('hidden');
    }

    // Soil Health Analyzer
    async analyzeSoilHealth(event) {
        event.preventDefault();
        
        const data = {
            nitrogen: parseFloat(document.getElementById('soil-n').value),
            phosphorus: parseFloat(document.getElementById('soil-p').value),
            potassium: parseFloat(document.getElementById('soil-k').value),
            ph: parseFloat(document.getElementById('soil-ph').value),
            organic_carbon: parseFloat(document.getElementById('soil-oc').value),
            soil_type: document.getElementById('soil-type').value
        };

        try {
            const response = await fetch('/api/ml/soil-health', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                this.renderSoilHealthResult(result.data);
            } else {
                alert('Failed to analyze soil health');
            }
        } catch (error) {
            console.error('Soil health analysis failed:', error);
            alert('Error analyzing soil health');
        }
    }

    renderSoilHealthResult(data) {
        const container = document.getElementById('soil-result');
        
        const healthColors = {
            poor: { bg: 'bg-red-100', text: 'text-red-700', bar: 'bg-red-500' },
            fair: { bg: 'bg-yellow-100', text: 'text-yellow-700', bar: 'bg-yellow-500' },
            good: { bg: 'bg-green-100', text: 'text-green-700', bar: 'bg-green-500' },
            excellent: { bg: 'bg-emerald-100', text: 'text-emerald-700', bar: 'bg-emerald-500' }
        };
        
        const colors = healthColors[data.category] || healthColors.fair;
        
        let html = `
            <div class="p-4 ${colors.bg} rounded-xl text-center">
                <p class="text-sm text-gray-600 mb-1">Soil Health Index</p>
                <p class="text-5xl font-bold ${colors.text}">${(data.soil_health_index || 0).toFixed(1)}</p>
                <p class="text-lg font-semibold ${colors.text} mt-1">${(data.category || 'Unknown').toUpperCase()}</p>
                <div class="w-full bg-gray-200 rounded-full h-4 mt-3">
                    <div class="${colors.bar} h-4 rounded-full transition-all duration-500" style="width: ${data.soil_health_index}%"></div>
                </div>
            </div>
        `;

        // Component Scores
        if (data.component_scores) {
            html += `
                <div class="p-4 bg-gray-50 rounded-xl">
                    <p class="font-semibold text-gray-700 mb-3">📊 Component Scores</p>
                    <div class="space-y-2">
            `;
            
            const componentLabels = {
                nitrogen_score: 'Nitrogen (N)',
                phosphorus_score: 'Phosphorus (P)',
                potassium_score: 'Potassium (K)',
                ph_score: 'pH Level',
                organic_carbon_score: 'Organic Carbon'
            };
            
            for (const [component, score] of Object.entries(data.component_scores)) {
                const pct = (score * 100).toFixed(0);
                html += `
                    <div class="flex items-center gap-2">
                        <span class="text-sm w-32 text-gray-600">${componentLabels[component] || component}</span>
                        <div class="flex-1 bg-gray-200 rounded-full h-2">
                            <div class="${pct >= 80 ? 'bg-green-500' : pct >= 50 ? 'bg-yellow-500' : 'bg-red-500'} h-2 rounded-full" style="width: ${pct}%"></div>
                        </div>
                        <span class="text-sm font-medium w-12 text-right">${pct}%</span>
                    </div>
                `;
            }
            
            html += `</div></div>`;
        }

        // Recommendations
        if (data.recommendations && data.recommendations.length > 0) {
            html += `
                <div class="p-4 bg-blue-50 rounded-xl">
                    <p class="font-semibold text-blue-800 mb-2">💡 Improvement Recommendations</p>
                    <ul class="text-sm text-blue-700 space-y-1">
                        ${data.recommendations.map(r => `<li>• ${r}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        // Suitable Crops
        if (data.suitable_crops && data.suitable_crops.length > 0) {
            html += `
                <div class="p-4 bg-green-50 rounded-xl">
                    <p class="font-semibold text-green-800 mb-2">🌾 Suitable Crops</p>
                    <div class="flex flex-wrap gap-2">
                        ${data.suitable_crops.map(crop => `<span class="px-3 py-1 bg-white rounded-full text-sm text-green-700">${this.getCropTranslation(crop)}</span>`).join('')}
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;
        container.classList.remove('hidden');
    }

    // Smart Irrigation Scheduler
    async getIrrigationSchedule(event) {
        event.preventDefault();
        
        const data = {
            crop: document.getElementById('irr-crop').value,
            growth_stage: document.getElementById('irr-stage').value,
            field_size: parseFloat(document.getElementById('irr-area').value),
            soil_type: document.getElementById('irr-soil').value,
            soil_moisture: parseFloat(document.getElementById('irr-moisture').value),
            temperature: 30,
            humidity: 60,
            wind_speed: 10,
            solar_radiation: 20,
            rainfall_forecast: [0, 0, 0, 0, 0, 0, 0]
        };

        // Try to get weather data for better predictions
        if (this.userData.latitude && this.userData.longitude) {
            try {
                const weatherResp = await fetch(`/api/weather?lat=${this.userData.latitude}&lon=${this.userData.longitude}`);
                const weatherData = await weatherResp.json();
                if (weatherData.current) {
                    data.temperature = weatherData.current.temperature;
                    data.humidity = weatherData.current.humidity;
                    data.wind_speed = weatherData.current.wind_speed;
                }
                if (weatherData.forecast) {
                    data.rainfall_forecast = weatherData.forecast.slice(0, 7).map(f => f.rainfall || 0);
                }
            } catch (e) {
                console.log('Using default weather data');
            }
        }

        try {
            const response = await fetch('/api/ml/irrigation-schedule', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                this.renderIrrigationResult(result.data);
            } else {
                alert('Failed to generate irrigation schedule');
            }
        } catch (error) {
            console.error('Irrigation scheduling failed:', error);
            alert('Error generating schedule');
        }
    }

    renderIrrigationResult(data) {
        const container = document.getElementById('irrigation-result');
        
        let html = `
            <div class="grid grid-cols-3 gap-3">
                <div class="p-3 bg-cyan-100 rounded-xl text-center">
                    <p class="text-xs text-gray-600">Total Water</p>
                    <p class="text-xl font-bold text-cyan-700">${(data.total_irrigation_liters / 1000).toFixed(0)}K L</p>
                </div>
                <div class="p-3 bg-blue-100 rounded-xl text-center">
                    <p class="text-xs text-gray-600">Depth</p>
                    <p class="text-xl font-bold text-blue-700">${data.total_irrigation_mm?.toFixed(1)} mm</p>
                </div>
                <div class="p-3 bg-indigo-100 rounded-xl text-center">
                    <p class="text-xs text-gray-600">Efficiency</p>
                    <p class="text-xl font-bold text-indigo-700">${((data.irrigation_efficiency || 0.5) * 100).toFixed(0)}%</p>
                </div>
            </div>
        `;

        // 7-Day Schedule
        if (data.schedule && data.schedule.length > 0) {
            html += `
                <div class="p-4 bg-gray-50 rounded-xl">
                    <p class="font-semibold text-gray-700 mb-3">📅 7-Day Irrigation Schedule</p>
                    <div class="overflow-x-auto">
                        <table class="w-full text-sm">
                            <thead>
                                <tr class="border-b">
                                    <th class="text-left py-2 px-1">Date</th>
                                    <th class="text-right py-2 px-1">Water</th>
                                    <th class="text-right py-2 px-1">Rain</th>
                                    <th class="text-right py-2 px-1">Moisture</th>
                                </tr>
                            </thead>
                            <tbody>
            `;
            
            data.schedule.forEach(day => {
                const date = new Date(day.date).toLocaleDateString('en-IN', { weekday: 'short', day: 'numeric' });
                const irrigationLiters = day.irrigation_liters || 0;
                const needsIrrigation = irrigationLiters > 0;
                
                html += `
                    <tr class="border-b ${needsIrrigation ? 'bg-cyan-50' : ''}">
                        <td class="py-2 px-1">
                            <span class="font-medium">${date}</span>
                            <span class="text-xs text-gray-500 block">${day.growth_stage || ''}</span>
                        </td>
                        <td class="text-right py-2 px-1 ${needsIrrigation ? 'text-cyan-700 font-bold' : 'text-gray-400'}">
                            ${needsIrrigation ? (irrigationLiters / 1000).toFixed(0) + 'K L' : '-'}
                        </td>
                        <td class="text-right py-2 px-1 text-blue-600">${day.rainfall?.toFixed(1) || 0} mm</td>
                        <td class="text-right py-2 px-1">${day.soil_moisture_percent?.toFixed(0) || '-'}%</td>
                    </tr>
                `;
            });
            
            html += `</tbody></table></div></div>`;
        }

        // Recommendations
        if (data.recommendations && data.recommendations.length > 0) {
            html += `
                <div class="p-4 bg-blue-50 rounded-xl">
                    <p class="font-semibold text-blue-800 mb-2">💡 Recommendations</p>
                    <ul class="text-sm text-blue-700 space-y-1">
                        ${data.recommendations.map(r => `<li>• ${r}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        container.innerHTML = html;
        container.classList.remove('hidden');
    }

    // Profit Calculator
    async calculateProfit(event) {
        event.preventDefault();
        
        const data = {
            crop: document.getElementById('profit-crop').value,
            state: this.userData.state || 'Karnataka',
            area_hectares: parseFloat(document.getElementById('profit-area').value),
            seed_cost: parseFloat(document.getElementById('profit-seed').value) || 0,
            fertilizer_cost: parseFloat(document.getElementById('profit-fert').value) || 0,
            pesticide_cost: parseFloat(document.getElementById('profit-pest').value) || 0,
            irrigation_cost: parseFloat(document.getElementById('profit-irr').value) || 0,
            labor_cost: parseFloat(document.getElementById('profit-labor').value) || 0,
            other_costs: parseFloat(document.getElementById('profit-other').value) || 0
        };

        try {
            const response = await fetch('/api/ml/profit-calculator', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                this.renderProfitResult(result.data);
            } else {
                alert('Failed to calculate profit');
            }
        } catch (error) {
            console.error('Profit calculation failed:', error);
            alert('Error calculating profit');
        }
    }

    renderProfitResult(data) {
        const container = document.getElementById('profit-result');
        
        const isProfit = data.financials?.net_profit >= 0;
        const verdictColor = isProfit ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700';
        
        let html = `
            <div class="p-4 ${verdictColor} rounded-xl text-center">
                <p class="text-sm opacity-75 mb-1">${data.verdict || 'Analysis'}</p>
                <p class="text-4xl font-bold">${isProfit ? '+' : ''}₹${Math.abs(data.financials?.net_profit || 0).toLocaleString()}</p>
                <p class="text-sm mt-1">Net ${isProfit ? 'Profit' : 'Loss'}</p>
            </div>
            
            <div class="grid grid-cols-2 gap-3">
                <div class="p-3 bg-green-50 rounded-xl text-center">
                    <p class="text-xs text-gray-600">Gross Revenue</p>
                    <p class="text-lg font-bold text-green-700">₹${(data.revenue?.gross_revenue || 0).toLocaleString()}</p>
                </div>
                <div class="p-3 bg-red-50 rounded-xl text-center">
                    <p class="text-xs text-gray-600">Total Cost</p>
                    <p class="text-lg font-bold text-red-700">₹${(data.total_cost || 0).toLocaleString()}</p>
                </div>
            </div>
        `;

        // Financial Metrics
        if (data.financials) {
            html += `
                <div class="p-4 bg-gray-50 rounded-xl">
                    <p class="font-semibold text-gray-700 mb-3">📈 Financial Metrics</p>
                    <div class="grid grid-cols-2 gap-2 text-sm">
                        <div class="bg-white p-2 rounded-lg">
                            <p class="text-xs text-gray-500">ROI</p>
                            <p class="font-bold ${data.financials.return_on_investment_percent >= 0 ? 'text-green-600' : 'text-red-600'}">${data.financials.return_on_investment_percent?.toFixed(1) || 0}%</p>
                        </div>
                        <div class="bg-white p-2 rounded-lg">
                            <p class="text-xs text-gray-500">Profit Margin</p>
                            <p class="font-bold ${data.financials.profit_margin_percent >= 0 ? 'text-green-600' : 'text-red-600'}">${data.financials.profit_margin_percent?.toFixed(1) || 0}%</p>
                        </div>
                        <div class="bg-white p-2 rounded-lg col-span-2">
                            <p class="text-xs text-gray-500">Break-even Yield</p>
                            <p class="font-bold text-gray-800">${(data.financials.break_even_yield_kg || 0).toLocaleString()} kg</p>
                        </div>
                    </div>
                </div>
            `;
        }

        // Yield Prediction
        if (data.yield) {
            html += `
                <div class="p-4 bg-amber-50 rounded-xl">
                    <p class="font-semibold text-amber-800 mb-2">🌾 Yield Estimate</p>
                    <div class="grid grid-cols-2 gap-2 text-sm">
                        <div>
                            <p class="text-xs text-gray-500">Predicted Yield/Ha</p>
                            <p class="font-bold">${(data.yield.predicted_kg_per_ha || 0).toLocaleString()} kg</p>
                        </div>
                        <div>
                            <p class="text-xs text-gray-500">Total Yield</p>
                            <p class="font-bold">${(data.yield.total_kg || 0).toLocaleString()} kg</p>
                        </div>
                    </div>
                    <p class="text-xs text-amber-600 mt-2">Confidence: ${(data.yield.confidence || 0).toFixed(0)}%</p>
                </div>
            `;
        }

        // Cost Breakdown
        if (data.costs) {
            html += `
                <div class="p-4 bg-white border rounded-xl">
                    <p class="font-semibold text-gray-700 mb-2">💸 Cost Breakdown</p>
                    <div class="space-y-1 text-sm">
            `;
            
            const costLabels = {
                seed: 'Seeds', fertilizer: 'Fertilizer', pesticides: 'Pesticides',
                irrigation: 'Irrigation', labor: 'Labor', machinery: 'Machinery',
                transport: 'Transport', other: 'Other'
            };
            
            for (const [cost, value] of Object.entries(data.costs)) {
                if (value > 0) {
                    html += `
                        <div class="flex justify-between">
                            <span class="text-gray-600">${costLabels[cost] || cost}</span>
                            <span class="font-medium">₹${value.toLocaleString()}</span>
                        </div>
                    `;
                }
            }
            
            html += `</div></div>`;
        }

        container.innerHTML = html;
        container.classList.remove('hidden');
    }
}

// Initialize app
const app = new CeresApp();

// Global functions
function changeLanguage(lang) {
    app.changeLanguage(lang);
}

function refreshWeather() {
    app.loadWeather();
    app.loadWeatherAlerts();
    app.loadAirQuality();
}

function refreshMarketPrices() {
    app.loadMarketPrices();
}

function filterMarketPrices() {
    app.filterMarketPrices();
}

function getCropRecommendation(event) {
    app.getCropRecommendation(event);
}

function updateRangeValue(id) {
    app.updateRangeValue(id);
}

function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Modal functions
function openModal(modalId) {
    document.getElementById(modalId).classList.add('active');
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

function openWaterCalculator() {
    openModal('water-modal');
}

function openFertilizerCalculator() {
    openModal('fertilizer-modal');
}

function openCropCalendar() {
    app.loadCropCalendar();
    openModal('calendar-modal');
}

function openFarmDiary() {
    app.loadFarmDiary();
    openModal('diary-modal');
}

function calculateWater(event) {
    app.calculateWater(event);
}

function calculateFertilizer(event) {
    app.calculateFertilizer(event);
}

function addDiaryEntry(event) {
    app.addDiaryEntry(event);
}

// ML Tools Modal Functions
function openDiseaseRiskAnalyzer() {
    openModal('disease-modal');
}

function openYieldPredictor() {
    openModal('yield-modal');
}

function openSoilHealthAnalyzer() {
    openModal('soil-modal');
}

function openIrrigationScheduler() {
    openModal('irrigation-modal');
}

function openProfitCalculator() {
    openModal('profit-modal');
}

function analyzeDiseaseRisk(event) {
    app.analyzeDiseaseRisk(event);
}

function predictYield(event) {
    app.predictYield(event);
}

function analyzeSoilHealth(event) {
    app.analyzeSoilHealth(event);
}

function getIrrigationSchedule(event) {
    app.getIrrigationSchedule(event);
}

function calculateProfit(event) {
    app.calculateProfit(event);
}

// Close modals on outside click
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.classList.remove('active');
    }
});

// Close modals on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal.active').forEach(modal => {
            modal.classList.remove('active');
        });
    }
});
