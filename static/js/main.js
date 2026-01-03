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
        
        // Sunrise/Sunset
        if (current.sunrise) {
            document.getElementById('sunrise').textContent = current.sunrise;
        }
        if (current.sunset) {
            document.getElementById('sunset').textContent = current.sunset;
        }
        
        // Weather icon - use emoji-based icons for Open-Meteo
        const iconEl = document.getElementById('weather-icon');
        const weatherCode = current.weather_code || 0;
        iconEl.src = this.getWeatherIcon(weatherCode);
        iconEl.alt = current.description || 'Weather';
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
            
            const weatherCode = day.weather_code || 0;
            const emoji = this.getWeatherEmoji(weatherCode);
            
            dayEl.innerHTML = `
                <p class="text-xs text-gray-500 mb-1">${dayName}</p>
                <span class="text-2xl">${emoji}</span>
                <p class="text-sm font-bold text-gray-900">${day.temp_max}°</p>
                <p class="text-xs text-gray-500">${day.temp_min}°</p>
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
            let url = '/api/air-quality';
            if (this.userData.latitude && this.userData.longitude) {
                url += `?lat=${this.userData.latitude}&lon=${this.userData.longitude}`;
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

        const aqi = data.aqi || data.pm25 || 0;
        aqiValue.textContent = Math.round(aqi);
        
        let label = 'Good';
        let color = 'text-green-600';
        if (aqi > 150) { label = 'Unhealthy'; color = 'text-red-600'; }
        else if (aqi > 100) { label = 'Moderate'; color = 'text-yellow-600'; }
        else if (aqi > 50) { label = 'Satisfactory'; color = 'text-green-500'; }
        
        aqiLabel.textContent = label;
        aqiValue.className = `text-4xl font-bold mb-2 ${color}`;
        
        if (data.pm25) pm25.textContent = data.pm25.toFixed(1) + ' µg/m³';
        if (data.pm10) pm10.textContent = data.pm10.toFixed(1) + ' µg/m³';
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
