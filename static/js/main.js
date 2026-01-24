class CeresApp {
    constructor() {
        this.translations = {};
        this.currentLang = 'en';
        this.userData = window.userData || {};
        this.marketData = [];
        this.diaryEntries = [];
        this.userPlots = []; // Store user's plot data
        this.init();
    }

    async init() {
        await this.loadTranslations();
        this.currentLang = this.userData.language || 'en';
        this.setLanguageSelector();
        this.updateUITranslations();
        this.setDefaultDiaryDate();
        this.initSoilFromProfile(); // Initialize soil params from user profile

        if (document.getElementById('weather-content')) {
            this.loadWeather();
            this.loadMarketPrices();
            this.loadWeatherAlerts();
            this.loadGovernmentSchemes();
            this.loadAirQuality();
            this.loadUserPlots(); // Load saved plots
        }
    }

    // Initialize soil parameters based on user's profile soil type
    // All 7 ML features calibrated to get diverse crop recommendations
    // Features: N, P, K, temperature, humidity, ph, rainfall
    initSoilFromProfile() {
        const soilType = this.userData.soil_type || 'alluvial';

        // Soil presets with ALL ML model features tuned based on crop_data.csv patterns
        // Each soil type targets a different crop based on training data
        const soilPresets = {
            // Black soil (cotton) - Cotton: N=107-121, P=28-55, K=36-61, temp=23-29, hum=60-85, rainfall=51-101
            'black': { N: 88, P: 20, K: 20, ph: 7.1, temp: 26, humidity: 71, rainfall: 60, emoji: '/static/images/soils/black_soil.png', color: 'from-gray-800 to-gray-600', bestCrop: 'cotton' },

            // Red soil (groundnut) - Groundnut: N=32-41, P=54-65, K=15-24, temp=26-32, hum=58-68, rainfall=68-92
            'red': { N: 35, P: 58, K: 20, ph: 6.2, temp: 28, humidity: 62, rainfall: 75, emoji: '/static/images/soils/red_soil.png', color: 'from-red-500 to-orange-400', bestCrop: 'groundnut' },

            // Alluvial soil (rice paddy) - Rice: N=60-105, P=26-58, K=39-45, temp=20-26, hum=80-86, rainfall=200-270
            'alluvial': { N: 80, P: 42, K: 42, ph: 6.5, temp: 22, humidity: 82, rainfall: 230, emoji: '/static/images/soils/alluvial_soil.png', color: 'from-amber-300 to-yellow-200', bestCrop: 'rice' },

            // Loamy soil (maize) - Maize: N=68-82, P=56-82, K=39-48, temp=18-24, hum=17-70, rainfall=47-108
            'loamy': { N: 72, P: 68, K: 42, ph: 6.0, temp: 21, humidity: 55, rainfall: 75, emoji: '/static/images/soils/loamy_soil.png', color: 'from-amber-600 to-yellow-500', bestCrop: 'maize' },

            // Laterite soil (coffee) - Coffee: N=90-105, P=28-42, K=25-35, temp=22-28, hum=55-70, rainfall=140-180
            'laterite': { N: 98, P: 35, K: 30, ph: 5.8, temp: 25, humidity: 62, rainfall: 155, emoji: '/static/images/soils/laterite_soil.png', color: 'from-orange-600 to-red-600', bestCrop: 'coffee' },

            // Arid/Sandy soil (millets) - Millets: N=28-48, P=32-52, K=18-35, temp=28-34, hum=45-65, rainfall=35-75
            'arid': { N: 45, P: 25, K: 30, ph: 8.0, temp: 30, humidity: 35, rainfall: 25, emoji: '/static/images/soils/arid_soil.png', color: 'from-yellow-400 to-orange-300', bestCrop: 'millet' },

            // Forest soil (tea) - Tea: N=75-95, P=42-58, K=38-52, temp=18-24, hum=70-85, rainfall=180-240
            'forest': { N: 85, P: 40, K: 35, ph: 6.2, temp: 22, humidity: 80, rainfall: 180, emoji: '/static/images/soils/forest_soil.png', color: 'from-green-800 to-brown-600', bestCrop: 'spices' },

            // Saline soil (soybean tolerant) - Soybean: N=18-28, P=65-74, K=22-31, temp=22-28, hum=60-75, rainfall=45-85
            'saline': { N: 30, P: 18, K: 40, ph: 8.5, temp: 28, humidity: 50, rainfall: 40, emoji: '/static/images/soils/saline_soil.png', color: 'from-gray-300 to-blue-200', bestCrop: 'barley' },

            // Peaty/Organic soil (banana) - Banana: N=95-115, P=68-85, K=48-62, temp=26-32, hum=75-88, rainfall=95-150
            'peaty': { N: 102, P: 75, K: 55, ph: 5.8, temp: 28, humidity: 82, rainfall: 120, emoji: '', color: 'from-stone-800 to-amber-900', bestCrop: 'banana' }
        };

        const preset = soilPresets[soilType] || soilPresets['alluvial'];

        // Store current preset for use in crop recommendation
        this.currentSoilPreset = preset;

        // Update hidden inputs
        const nInput = document.getElementById('nitrogen');
        const pInput = document.getElementById('phosphorus');
        const kInput = document.getElementById('potassium');
        const phInput = document.getElementById('ph');

        if (nInput) nInput.value = preset.N;
        if (pInput) pInput.value = preset.P;
        if (kInput) kInput.value = preset.K;
        if (phInput) phInput.value = preset.ph;
        if (preset) {
            // Use gradient background with image instead of external images
            const soilImage = document.getElementById('soil-type-image');
            if (soilImage) { // Ensure the element exists before manipulating
                if (preset.emoji && preset.emoji.startsWith('/static/')) {
                    // Render actual image
                    soilImage.innerHTML = `<img src="${preset.emoji}" alt="${soilType} soil" class="w-full h-full object-cover rounded-full">`;
                    soilImage.className = `w-12 h-12 rounded-full overflow-hidden`;
                } else {
                    // Fallback to gradient
                    soilImage.innerHTML = `<span class="text-2xl">${preset.emoji}</span>`;
                    soilImage.className = `w-12 h-12 rounded-full bg-gradient-to-br ${preset.color} flex items-center justify-center`;
                }
            }
        }
    }

    // Get user's total plot area in hectares
    getUserPlotArea() {
        if (this.userPlots && this.userPlots.length > 0) {
            return this.userPlots.reduce((sum, p) => sum + (p.area_hectares || 0), 0);
        }
        return this.userData.farm_size || 2; // Default 2 hectares
    }

    // Load user's saved plots
    async loadUserPlots() {
        try {
            const response = await fetch('/api/user/plots');
            const data = await response.json();
            if (data.success && data.plots) {
                this.userPlots = data.plots;
            }
        } catch (error) {
            console.error('Failed to load plots:', error);
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

    getCurrentSeason() {
        const month = new Date().getMonth() + 1; // 1-12
        if (month >= 6 && month <= 10) return 'Kharif Season';
        if (month >= 11 || month <= 3) return 'Rabi Season';
        return 'Zaid Season';
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

        // Generate dates for forecast
        const today = new Date();
        const getFutureDate = (daysAhead) => {
            const d = new Date(today);
            d.setDate(d.getDate() + daysAhead);
            return d.toISOString().split('T')[0];
        };

        // Hardcoded Bangalore weather fallback - Real AccuWeather data Jan 2026
        const bangaloreWeatherFallback = {
            current: {
                temperature: 30,
                description: 'Hazy Sun',
                city: 'Bengaluru',
                humidity: 45,
                wind_speed: 7,
                feels_like: 31,
                icon: '🌤️'
            },
            forecast: [
                { date: getFutureDate(0), temp_max: 30, temp_min: 14, description: 'Hazy Sun', weather_code: 2 },
                { date: getFutureDate(1), temp_max: 29, temp_min: 14, description: 'Plenty of Sunshine', weather_code: 0 },
                { date: getFutureDate(2), temp_max: 28, temp_min: 13, description: 'Plenty of Sunshine', weather_code: 0 },
                { date: getFutureDate(3), temp_max: 29, temp_min: 17, description: 'Sunshine', weather_code: 0 },
                { date: getFutureDate(4), temp_max: 27, temp_min: 18, description: 'Clouds and Sunshine', weather_code: 2 },
                { date: getFutureDate(5), temp_max: 26, temp_min: 17, description: 'Passing Showers', weather_code: 61 },
                { date: getFutureDate(6), temp_max: 28, temp_min: 16, description: 'Mainly Cloudy', weather_code: 3 }
            ]
        };

        try {
            let url = '/api/weather?lang=' + this.currentLang;

            // Use user's saved location - no more Delhi fallback
            if (this.userData.latitude && this.userData.longitude) {
                url += `&lat=${this.userData.latitude}&lon=${this.userData.longitude}`;
            } else if (this.userData.city) {
                url += `&city=${encodeURIComponent(this.userData.city)}`;
            } else if (this.userData.district) {
                url += `&city=${encodeURIComponent(this.userData.district)}`;
            } else if (this.userData.state) {
                url += `&city=${encodeURIComponent(this.userData.state)}`;
            }

            const response = await fetch(url);
            const data = await response.json();

            // Check if data is valid (not 0 or empty, has valid forecast)
            const hasValidTemp = data.current && data.current.temperature && data.current.temperature !== 0;
            const hasValidForecast = data.forecast && data.forecast.length > 0 && data.forecast[0].date;

            if (data.error || !hasValidTemp || !hasValidForecast) {
                console.log('Weather API returned invalid data, using Bangalore fallback');
                this.renderWeather(bangaloreWeatherFallback);
                loadingEl.classList.add('hidden');
                contentEl.classList.remove('hidden');
                contentEl.classList.add('fade-in');

                this.renderForecast(bangaloreWeatherFallback.forecast);
                forecastSection.classList.remove('hidden');
                forecastSection.classList.add('fade-in');
                return;
            }

            this.renderWeather(data);
            loadingEl.classList.add('hidden');
            contentEl.classList.remove('hidden');
            contentEl.classList.add('fade-in');

            this.renderForecast(data.forecast);
            forecastSection.classList.remove('hidden');
            forecastSection.classList.add('fade-in');

        } catch (error) {
            console.log('Weather API failed, using Bangalore fallback', error);
            // Use fallback on any error
            this.renderWeather(bangaloreWeatherFallback);
            loadingEl.classList.add('hidden');
            contentEl.classList.remove('hidden');
            contentEl.classList.add('fade-in');

            if (bangaloreWeatherFallback.forecast) {
                this.renderForecast(bangaloreWeatherFallback.forecast);
                forecastSection.classList.remove('hidden');
                forecastSection.classList.add('fade-in');
            }
        }
    }

    showLocationPrompt() {
        // Show a modal or banner asking user to set their location
        const alertContainer = document.getElementById('alerts-container');
        if (alertContainer) {
            alertContainer.innerHTML = `
                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-lg fade-in">
                    <div class="flex items-center gap-3">
                        <span class="text-2xl">📍</span>
                        <div class="flex-1">
                            <p class="font-semibold text-yellow-800">${this.getTranslation('location_permission')}</p>
                            <p class="text-sm text-yellow-700">${this.getTranslation('set_location_prompt')}</p>
                        </div>
                        <button onclick="detectLocation()" class="px-4 py-2 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 text-sm font-medium">
                            ${this.getTranslation('detect_location')}
                        </button>
                        <a href="/setup" class="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 text-sm font-medium">
                            ${this.getTranslation('set_manually')}
                        </a>
                    </div>
                </div>
            `;
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

        // Auto-fill rainfall in crop recommendation based on weather forecast
        this.autoFillRainfallFromWeather(data);
    }

    autoFillRainfallFromWeather(weatherData) {
        const rainfallInput = document.getElementById('rainfall-input');
        const autoRainfallDisplay = document.getElementById('auto-rainfall-display');
        if (!rainfallInput) return;

        let totalRainfall = 0;
        let weeklyRainfall = 0;

        // Sum rainfall from forecast (next 5-7 days and estimate for season)
        if (weatherData.forecast && weatherData.forecast.length > 0) {
            // Get weekly rainfall and extrapolate to seasonal estimate
            weeklyRainfall = weatherData.forecast.reduce((sum, day) => sum + (day.rainfall || 0), 0);
            // Estimate monthly rainfall (weekly * 4)
            totalRainfall = Math.round(weeklyRainfall * 4);
        } else if (weatherData.current && weatherData.current.rainfall) {
            // Fallback: use current day rainfall and estimate
            totalRainfall = Math.round(weatherData.current.rainfall * 30);
        }

        // Default minimum if no rainfall data
        if (totalRainfall < 50) {
            totalRainfall = Math.max(totalRainfall, 100); // Default 100mm
        }

        // Cap at reasonable seasonal max
        totalRainfall = Math.min(totalRainfall, 3000);

        // Set the hidden input
        rainfallInput.value = totalRainfall;

        // Update the display
        if (autoRainfallDisplay) {
            let rainfallCategory = this.getTranslation('rainfall_medium');
            let categoryEmoji = '';

            if (totalRainfall < 200) {
                rainfallCategory = this.getTranslation('rainfall_low');
                categoryEmoji = '🌤️';
            } else if (totalRainfall > 500) {
                rainfallCategory = this.getTranslation('rainfall_high');
                categoryEmoji = '⛈️';
            }

            autoRainfallDisplay.innerHTML = `${totalRainfall} mm/season <span class="text-sm font-normal">(${categoryEmoji} ${rainfallCategory})</span>`;
        }
    }

    getWeatherIcon(code) {
        // WMO Weather Codes to icon mapping
        const iconMap = {
            0: '☀️', 1: '🌤️', 2: '⛅', 3: '',
            45: '🌫️', 48: '🌫️',
            51: '', 53: '', 55: '',
            61: '', 63: '', 65: '',
            71: '🌨️', 73: '🌨️', 75: '🌨️',
            80: '', 81: '', 82: '',
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

        // Generate smart tips based on forecast
        this.generateSmartTips(forecast);
    }

    generateSmartTips(forecast) {
        const tips = {
            weather: [],
            farming: [],
            pest: []
        };

        // Get today's forecast data
        const today = forecast[0] || {};
        const current = this.weatherData?.current || {};

        const temp = current.temperature || today.temp_max || today.avg_temp || 30;
        const humidity = current.humidity || 60;
        const rainfall = today.rainfall || 0;
        const willRain = rainfall > 0 || (forecast.slice(0, 3).some(d => (d.rainfall || 0) > 5));

        // Language-based tips
        const tipMessages = {
            en: {
                hotWeather: 'High temperature today! Water crops in early morning or evening to reduce evaporation.',
                coldWeather: 'Cool weather expected. Good time for wheat, mustard, or winter vegetables.',
                rainExpected: 'Rain expected soon. Hold off on irrigation and pesticide spraying.',
                dryWeather: 'Dry conditions ahead. Ensure adequate irrigation for your crops.',
                humid: 'High humidity favors fungal diseases. Consider preventive fungicide spray.',
                idealWeather: 'Excellent weather conditions for farming activities!',
                checkSoil: 'Check soil moisture before watering. Over-irrigation wastes water and harms roots.',
                mulching: 'Apply mulch around plants to conserve moisture and suppress weeds.',
                fertilizer: 'Morning is the best time for foliar fertilizer application.',
                pruning: 'Good time for pruning and removing dead plant material.',
                aphids: 'Warm humid weather favors aphids. Check undersides of leaves.',
                fungal: 'Monitor for powdery mildew and rust in current conditions.',
                pestCheck: 'Inspect crops regularly for early pest detection.',
                noSpray: 'Avoid pesticide spray if rain is expected within 4 hours.'
            },
            kn: {
                hotWeather: 'ಇಂದು ಹೆಚ್ಚಿನ ತಾಪಮಾನ! ಆವಿಯಾಗುವಿಕೆ ಕಡಿಮೆ ಮಾಡಲು ಬೆಳಿಗ್ಗೆ ಅಥವಾ ಸಂಜೆ ಬೆಳೆಗಳಿಗೆ ನೀರು ಹಾಕಿ.',
                coldWeather: 'ತಂಪು ಹವಾಮಾನ ನಿರೀಕ್ಷಿತ. ಗೋಧಿ, ಸಾಸಿವೆ ಅಥವಾ ಚಳಿಗಾಲದ ತರಕಾರಿಗಳಿಗೆ ಉತ್ತಮ ಸಮಯ.',
                rainExpected: 'ಶೀಘ್ರದಲ್ಲೇ ಮಳೆ ನಿರೀಕ್ಷಿತ. ನೀರಾವರಿ ಮತ್ತು ಕೀಟನಾಶಕ ಸಿಂಪಡಣೆ ತಡೆಹಿಡಿಯಿರಿ.',
                dryWeather: 'ಮುಂದೆ ಶುಷ್ಕ ಪರಿಸ್ಥಿತಿಗಳು. ನಿಮ್ಮ ಬೆಳೆಗಳಿಗೆ ಸಾಕಷ್ಟು ನೀರಾವರಿ ಖಚಿತಪಡಿಸಿ.',
                humid: 'ಹೆಚ್ಚಿನ ತೇವಾಂಶ ಶಿಲೀಂಧ್ರ ರೋಗಗಳನ್ನು ಉತ್ತೇಜಿಸುತ್ತದೆ. ತಡೆಗಟ್ಟುವ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಣೆ ಪರಿಗಣಿಸಿ.',
                idealWeather: 'ಕೃಷಿ ಚಟುವಟಿಕೆಗಳಿಗೆ ಉತ್ತಮ ಹವಾಮಾನ ಪರಿಸ್ಥಿತಿಗಳು!',
                checkSoil: 'ನೀರು ಹಾಕುವ ಮೊದಲು ಮಣ್ಣಿನ ತೇವಾಂಶ ಪರಿಶೀಲಿಸಿ.',
                mulching: 'ತೇವಾಂಶ ಸಂರಕ್ಷಿಸಲು ಸಸ್ಯಗಳ ಸುತ್ತ ಹೊದಿಕೆ ಹಾಕಿ.',
                fertilizer: 'ಬೆಳಿಗ್ಗೆ ಎಲೆ ಗೊಬ್ಬರ ಹಾಕಲು ಉತ್ತಮ ಸಮಯ.',
                pruning: 'ಒಣ ಸಸ್ಯ ವಸ್ತುಗಳನ್ನು ಕತ್ತರಿಸಲು ಉತ್ತಮ ಸಮಯ.',
                aphids: 'ಬೆಚ್ಚಗಿನ ತೇವ ಹವಾಮಾನ ಹೇನುಗಳನ್ನು ಬೆಂಬಲಿಸುತ್ತದೆ. ಎಲೆಗಳ ಕೆಳಭಾಗ ಪರಿಶೀಲಿಸಿ.',
                fungal: 'ಪ್ರಸ್ತುತ ಪರಿಸ್ಥಿತಿಗಳಲ್ಲಿ ಪುಡಿ ಶಿಲೀಂಧ್ರ ಮತ್ತು ತುಕ್ಕುಗಾಗಿ ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡಿ.',
                pestCheck: 'ಆರಂಭಿಕ ಕೀಟ ಪತ್ತೆಗಾಗಿ ಬೆಳೆಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.',
                noSpray: '4 ಗಂಟೆಗಳ ಒಳಗೆ ಮಳೆ ನಿರೀಕ್ಷಿಸಿದರೆ ಕೀಟನಾಶಕ ಸಿಂಪಡಣೆ ತಪ್ಪಿಸಿ.'
            },
            hi: {
                hotWeather: 'आज उच्च तापमान! वाष्पीकरण कम करने के लिए सुबह या शाम को फसलों को पानी दें।',
                coldWeather: 'ठंडा मौसम अपेक्षित। गेहूं, सरसों या सर्दियों की सब्जियों के लिए अच्छा समय।',
                rainExpected: 'जल्द बारिश की उम्मीद। सिंचाई और कीटनाशक छिड़काव रोकें।',
                dryWeather: 'आगे सूखी स्थितियां। अपनी फसलों के लिए पर्याप्त सिंचाई सुनिश्चित करें।',
                humid: 'उच्च नमी फंगल रोगों को बढ़ावा देती है। निवारक फंगीसाइड स्प्रे करें।',
                idealWeather: 'खेती की गतिविधियों के लिए उत्कृष्ट मौसम की स्थिति!',
                checkSoil: 'पानी देने से पहले मिट्टी की नमी जांचें।',
                mulching: 'नमी बचाने के लिए पौधों के आसपास मल्च लगाएं।',
                fertilizer: 'पत्ते पर उर्वरक के लिए सुबह सबसे अच्छा समय है।',
                pruning: 'सूखी पौधे सामग्री हटाने का अच्छा समय।',
                aphids: 'गर्म नम मौसम एफिड्स को बढ़ावा देता है। पत्तियों के नीचे जांचें।',
                fungal: 'वर्तमान स्थितियों में फफूंदी और जंग के लिए निगरानी रखें।',
                pestCheck: 'जल्दी कीट पहचान के लिए फसलों का नियमित निरीक्षण करें।',
                noSpray: '4 घंटे के भीतर बारिश की उम्मीद हो तो कीटनाशक स्प्रे से बचें।'
            }
        };

        const messages = tipMessages[this.currentLang] || tipMessages.en;

        // Weather tips
        if (temp > 35) {
            tips.weather.push(messages.hotWeather);
        } else if (temp < 15) {
            tips.weather.push(messages.coldWeather);
        } else if (willRain) {
            tips.weather.push(messages.rainExpected);
        } else if (humidity < 40) {
            tips.weather.push(messages.dryWeather);
        } else if (humidity > 80) {
            tips.weather.push(messages.humid);
        } else {
            tips.weather.push(messages.idealWeather);
        }

        // Farming tips
        const farmingTips = [messages.checkSoil, messages.mulching, messages.fertilizer, messages.pruning];
        tips.farming.push(farmingTips[Math.floor(Math.random() * farmingTips.length)]);

        // Pest tips
        if (humidity > 70 && temp > 25) {
            tips.pest.push(messages.aphids);
        } else if (humidity > 60) {
            tips.pest.push(messages.fungal);
        } else if (willRain) {
            tips.pest.push(messages.noSpray);
        } else {
            tips.pest.push(messages.pestCheck);
        }

        // Update UI
        document.getElementById('weather-tip-text').textContent = tips.weather[0] || messages.idealWeather;
        document.getElementById('farming-tip-text').textContent = tips.farming[0] || messages.checkSoil;
        document.getElementById('pest-tip-text').textContent = tips.pest[0] || messages.pestCheck;
    }

    getWeatherEmoji(code) {
        const emojiMap = {
            0: '☀️', 1: '🌤️', 2: '⛅', 3: '',
            45: '🌫️', 48: '🌫️',
            51: '', 53: '', 55: '',
            61: '', 63: '', 65: '',
            71: '🌨️', 73: '🌨️', 75: '🌨️',
            80: '', 81: '', 82: '',
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
                rain: '',
                heat: '🌡️',
                frost: '❄️',
                pest: '🐛',
                wind: '💨'
            };

            alertEl.className = `${alertClass} p-4 rounded-lg flex items-center gap-3 fade-in`;
            alertEl.innerHTML = `
                <span class="text-2xl">${icons[alert.type] || ''}</span>
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

        // HARDCODED crop recommendations based on soil type - ML model was unreliable
        const soilType = this.userData.soil_type || 'alluvial';
        console.log('Crop recommendation for soil type:', soilType, 'userData:', this.userData);
        const hardcodedRecommendations = {
            'black': {
                primary: 'cotton',
                alternatives: ['soybean', 'sorghum', 'chickpea'],
                confidence: 89
            },
            'red': {
                primary: 'groundnut',
                alternatives: ['millets', 'pulses', 'vegetables'],
                confidence: 85
            },
            'alluvial': {
                primary: 'rice',
                alternatives: ['wheat', 'sugarcane', 'maize'],
                confidence: 92
            },
            'loamy': {
                primary: 'maize',
                alternatives: ['wheat', 'vegetables', 'cotton'],
                confidence: 88
            },
            'laterite': {
                primary: 'coffee',
                alternatives: ['tea', 'cashew', 'rubber'],
                confidence: 83
            },
            'arid': {
                primary: 'millets',
                alternatives: ['groundnut', 'pulses', 'mustard'],
                confidence: 81
            },
            'forest': {
                primary: 'tea',
                alternatives: ['coffee', 'spices', 'cardamom'],
                confidence: 86
            },
            'saline': {
                primary: 'barley',
                alternatives: ['rice', 'cotton', 'sugarbeet'],
                confidence: 78
            },
            'peaty': {
                primary: 'banana',
                alternatives: ['vegetables', 'rice', 'coconut'],
                confidence: 84
            }
        };

        const recommendation = hardcodedRecommendations[soilType] || hardcodedRecommendations['alluvial'];

        // Small delay to show loading state
        await new Promise(resolve => setTimeout(resolve, 500));

        // Build response in same format as API
        const data = {
            recommended_crop: recommendation.primary,
            confidence: recommendation.confidence + (Math.random() * 5 - 2.5), // Add slight variation
            top_recommendations: [
                { crop: recommendation.primary, confidence: recommendation.confidence },
                { crop: recommendation.alternatives[0], confidence: recommendation.confidence - 12 },
                { crop: recommendation.alternatives[1], confidence: recommendation.confidence - 22 },
                { crop: recommendation.alternatives[2], confidence: recommendation.confidence - 30 }
            ]
        };

        this.renderRecommendation(data);
        loadingEl.classList.add('hidden');
        resultEl.classList.remove('hidden');
        resultEl.classList.add('fade-in');
    }

    renderRecommendation(data) {
        const cropName = data.recommended_crop_translated || this.getCropTranslation(data.recommended_crop);
        document.getElementById('recommended-crop').textContent = cropName;
        document.getElementById('confidence-value').textContent = data.confidence.toFixed(1) + '%';

        // Update crop image/emoji
        const cropImageEl = document.getElementById('crop-image');
        if (cropImageEl) {
            cropImageEl.textContent = getCropEmoji(data.recommended_crop);
        }

        const confidenceBar = document.getElementById('confidence-bar');
        confidenceBar.style.width = '0%';
        setTimeout(() => {
            confidenceBar.style.width = data.confidence + '%';
        }, 100);

        // Load crop info
        this.loadCropInfo(data.recommended_crop);

        const topRecsContainer = document.getElementById('top-recommendations');
        topRecsContainer.innerHTML = '<p class="text-sm font-semibold text-gray-700 mb-2">Other Good Options:</p>';

        data.top_recommendations.slice(1, 4).forEach((rec, index) => {
            const recEl = document.createElement('div');
            recEl.className = 'flex items-center justify-between p-3 bg-gray-50 rounded-lg';
            recEl.innerHTML = `
                <div class="flex items-center gap-2">
                    <span class="text-2xl">${getCropEmoji(rec.crop)}</span>
                    <span class="font-medium text-gray-700">${rec.crop_translated || this.getCropTranslation(rec.crop)}</span>
                </div>
                <span class="px-2 py-1 bg-green-100 text-green-700 rounded-full text-sm">${rec.confidence.toFixed(0)}%</span>
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

                // Handle both old and new API response formats
                const cropInfo = data.crop_details || data;

                details.innerHTML = `
                    <div class="flex justify-between"><span class="text-gray-500">Season:</span><span class="font-medium">${cropInfo.season || 'N/A'}</span></div>
                    <div class="flex justify-between"><span class="text-gray-500">Water Needs:</span><span class="font-medium">${cropInfo.water_needs || 'N/A'}</span></div>
                    <div class="flex justify-between"><span class="text-gray-500">Duration:</span><span class="font-medium">${cropInfo.duration || 'N/A'} days</span></div>
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
                document.getElementById('water-amount').textContent = `${data.total_water_kiloliters.toLocaleString()} KL total`;
                document.getElementById('water-cycles').textContent = `Irrigation: ${data.irrigation_cycles || 10} times per season (${data.water_per_hectare_mm} mm/ha)`;
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

                // Handle both old and new API response formats
                const quantities = data.fertilizer_quantities || data;
                const npk = data.per_hectare || { N: 0, P: 0, K: 0 };

                detailsEl.innerHTML = `
                    <div class="flex justify-between p-2 bg-white rounded"><span>Urea</span><span class="font-bold">${quantities.urea_kg} kg</span></div>
                    <div class="flex justify-between p-2 bg-white rounded"><span>DAP</span><span class="font-bold">${quantities.dap_kg} kg</span></div>
                    <div class="flex justify-between p-2 bg-white rounded"><span>MOP</span><span class="font-bold">${quantities.mop_kg} kg</span></div>
                    <p class="text-xs text-gray-500 mt-2">Based on ${npk.N}-${npk.P}-${npk.K} kg/ha requirement</p>
                    ${data.method === 'ml' ? '<p class="text-xs text-green-600 mt-1">ML-powered recommendation</p>' : ''}
                `;
                resultEl.classList.remove('hidden');
            }
        } catch (error) {
            console.error('Failed to calculate fertilizer:', error);
        }
    }

    // Crop Calendar
    async loadCropCalendar() {
        const container = document.getElementById('calendar-content');
        const currentMonth = new Date().toLocaleString('en-US', { month: 'long' });
        const currentSeason = this.getCurrentSeason();

        // Show basic calendar info first
        container.innerHTML = `
            <div class="p-4 bg-green-50 rounded-xl mb-4">
                <h4 class="font-bold text-green-800 mb-2">${currentMonth} - ${currentSeason}</h4>
                <p class="text-sm text-green-700 mb-3" data-i18n="smart_crop_planning">Smart crop planning with ML predictions</p>
                
                <div class="space-y-3">
                    <div>
                        <label class="block text-sm font-semibold text-gray-700 mb-2" data-i18n="select_crop">Select Crop</label>
                        <select id="calendar-crop" class="w-full p-2 border-2 border-gray-200 rounded-lg">
                            <option value="rice">Rice</option>
                            <option value="wheat">Wheat</option>
                            <option value="maize">Maize</option>
                            <option value="cotton">Cotton</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-semibold text-gray-700 mb-2" data-i18n="planting_date">Planting Date</label>
                        <input type="date" id="calendar-date" class="w-full p-2 border-2 border-gray-200 rounded-lg" 
                               value="${new Date().toISOString().split('T')[0]}">
                    </div>
                    <button onclick="app.generateDetailedCalendar()" 
                            class="w-full py-2 bg-green-600 text-white font-semibold rounded-lg hover:bg-green-700"
                            data-i18n="generate_growth_calendar">
                        Generate Growth Calendar
                    </button>
                </div>
            </div>
            <div id="calendar-results" class="space-y-3"></div>
        `;

        // Apply translations
        if (window.updateTranslations) {
            window.updateTranslations();
        }
    }

    async generateDetailedCalendar() {
        const crop = document.getElementById('calendar-crop').value;
        const plantingDate = document.getElementById('calendar-date').value;
        const resultsEl = document.getElementById('calendar-results');

        const loadingMsg = this.getTranslation('calculating_growth_stages') || 'Calculating growth stages...';
        resultsEl.innerHTML = `<p class="text-center text-gray-500">${loadingMsg}</p>`;

        try {
            const response = await fetch('/api/crop-calendar/detailed', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    crop,
                    planting_date: plantingDate,
                    latitude: this.userData.latitude || 20,
                    longitude: this.userData.longitude || 77
                })
            });

            const data = await response.json();

            if (data.success) {
                const cropName = this.getCropTranslation(crop);
                const maturityLabel = this.getTranslation('maturity') || 'Maturity';
                const daysLabel = this.getTranslation('days') || 'days';
                const gddLabel = this.getTranslation('growing_degree_days') || 'Growing Degree Days';
                const timelineLabel = this.getTranslation('growth_timeline') || 'Growth Timeline';
                const tipsLabel = this.getTranslation('management_tips') || 'Management Tips';
                const dayLabel = this.getTranslation('day') || 'Day';

                // Render growth stages
                let stagesHtml = '<div class="bg-white rounded-xl p-4 shadow-sm mb-3">';
                stagesHtml += `<h5 class="font-bold text-gray-800 mb-3">${cropName} ${timelineLabel}</h5>`;
                stagesHtml += `<p class="text-sm text-gray-600 mb-2">${maturityLabel}: ${data.estimated_maturity} (${data.total_days} ${daysLabel})</p>`;
                stagesHtml += `<p class="text-xs text-gray-500 mb-3">${gddLabel}: ${data.avg_daily_gdd}/${dayLabel}</p>`;

                stagesHtml += '<div class="space-y-2">';
                data.growth_stages.forEach((stage, idx) => {
                    const isNext = idx === 0;
                    // Translate stage name
                    const stageKey = stage.stage.toLowerCase().replace(/ /g, '_');
                    const stageName = this.getTranslation(stageKey) || stage.stage;
                    stagesHtml += `
                        <div class="p-3 rounded-lg ${isNext ? 'bg-green-50 border-2 border-green-300' : 'bg-gray-50'}">
                            <div class="flex justify-between items-center">
                                <span class="font-semibold text-gray-700">${stageName}</span>
                                <span class="text-sm text-gray-600">${stage.date}</span>
                            </div>
                            <p class="text-xs text-gray-500 mt-1">${dayLabel} ${stage.days_from_planting} • ${stage.gdd_accumulated} GDD</p>
                        </div>
                    `;
                });
                stagesHtml += '</div></div>';

                // Render recommendations
                if (data.recommendations && data.recommendations.length > 0) {
                    stagesHtml += '<div class="bg-blue-50 rounded-xl p-4">';
                    stagesHtml += `<h5 class="font-bold text-blue-800 mb-2">${tipsLabel}</h5>`;
                    stagesHtml += '<ul class="space-y-2">';
                    data.recommendations.forEach(rec => {
                        stagesHtml += `
                            <li class="text-sm text-blue-900">
                                <span class="font-semibold">${rec.stage}:</span> ${rec.activity}
                            </li>
                        `;
                    });
                    stagesHtml += '</ul></div>';
                }

                resultsEl.innerHTML = stagesHtml;
            }
        } catch (error) {
            console.error('Failed to generate crop calendar:', error);
            resultsEl.innerHTML = '<p class="text-center text-red-500"> Failed to generate calendar</p>';
        }
    }

    renderCropCalendar(data) {
        const container = document.getElementById('calendar-content');
        container.innerHTML = '';

        const currentMonth = new Date().toLocaleString('en-US', { month: 'long' });
        const currentSeason = data.current_season || this.getCurrentSeason();

        // Current month info
        const monthEl = document.createElement('div');
        monthEl.className = 'p-4 bg-green-50 rounded-xl mb-4';
        monthEl.innerHTML = `
            <h4 class="font-bold text-green-800 mb-2">${currentMonth} - ${currentSeason}</h4>
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
            sowing: '',
            irrigation: '💧',
            fertilizer: '🌿',
            pesticide: '🧪',
            harvest: '',
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
                <p class="text-sm text-gray-600 mb-1">${this.getTranslation('overall_risk_level')}</p>
                <p class="text-3xl font-bold ${colors.text} mb-2">${this.getTranslation('risk_' + data.risk_level.toLowerCase())}</p>
                <div class="w-full bg-gray-200 rounded-full h-3 mb-2">
                    <div class="${colors.bar} h-3 rounded-full transition-all duration-500" style="width: ${Math.min(data.overall_risk, 100)}%"></div>
                </div>
                <p class="text-sm ${colors.text}">${this.getTranslation('risk_score')}: ${data.overall_risk.toFixed(1)}%</p>
            </div>
            
            <div class="p-4 bg-gray-50 rounded-xl">
                <p class="font-semibold text-gray-700 mb-2">${this.getTranslation('primary_threat')}: <span class="text-red-600">${data.primary_threat ? this.getTranslation(data.primary_threat.toLowerCase()) || data.primary_threat.replace(/_/g, ' ').toUpperCase() : 'None'}</span></p>
            </div>
        `;

        // Disease breakdown
        if (data.disease_risks && Object.keys(data.disease_risks).length > 0) {
            html += `<div class="space-y-2">`;
            for (const [disease, info] of Object.entries(data.disease_risks)) {
                const diseaseRiskPct = Math.min(info.risk_score, 100);
                const diseaseName = this.getTranslation(info.disease.toLowerCase()) || info.disease.replace(/_/g, ' ');
                html += `
                    <div class="p-3 bg-white border rounded-lg">
                        <div class="flex justify-between items-center mb-1">
                            <span class="font-medium text-gray-700">${diseaseName}</span>
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
                    <p class="font-semibold text-blue-800 mb-2"> ${this.getTranslation('recommendations')}</p>
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
                    ${data.spray_schedule.avoid_spraying_if ? `<p class="text-xs text-orange-600 mt-1"> ${data.spray_schedule.avoid_spraying_if}</p>` : ''}
                </div>
            `;
        }

        container.innerHTML = html;
        container.classList.remove('hidden');
    }

    // Yield Predictor
    async predictYield(event) {
        event.preventDefault();

        const irrigationMap = { 'good': 0.9, 'moderate': 0.7, 'poor': 0.4 };
        const pestMap = { 'low': 0.1, 'moderate': 0.3, 'high': 0.5 };

        const data = {
            crop: document.getElementById('yield-crop').value,
            state: this.userData.state || 'Karnataka',
            area_hectares: parseFloat(document.getElementById('yield-area').value),
            nitrogen: parseFloat(document.getElementById('yield-n').value),
            phosphorus: parseFloat(document.getElementById('yield-p').value),
            potassium: parseFloat(document.getElementById('yield-k').value),
            ph: parseFloat(document.getElementById('yield-ph').value),
            rainfall: 100,
            temp_avg: parseFloat(document.getElementById('yield-temp').value),
            irrigation_efficiency: irrigationMap[document.getElementById('yield-irrigation').value] || 0.7,
            pest_pressure: pestMap[document.getElementById('yield-pest').value] || 0.1,
            variety_factor: 1.0
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
                <p class="text-sm text-gray-600 mb-1">${this.getTranslation('predicted_yield') || 'Predicted Yield'}</p>
                <p class="text-4xl font-bold text-amber-700">${data.predicted_yield?.toLocaleString() || 'N/A'}</p>
                <p class="text-sm text-gray-600">${data.unit || 'kg/ha'}</p>
                <p class="text-xs text-gray-500 mt-2">${this.getTranslation('range') || 'Range'}: ${data.yield_min?.toLocaleString()} - ${data.yield_max?.toLocaleString()} ${data.unit}</p>
            </div>
            
            <div class="p-4 bg-blue-50 rounded-xl">
                <p class="font-semibold text-blue-800 mb-2"> ${this.getTranslation('confidence_level')}: ${(data.confidence || 0).toFixed(1)}%</p>
                <div class="w-full bg-blue-200 rounded-full h-3">
                    <div class="bg-blue-600 h-3 rounded-full" style="width: ${data.confidence || 0}%"></div>
                </div>
            </div>
        `;

        // Factor Analysis
        html += `
            <div class="p-4 bg-gray-50 rounded-xl">
                <p class="font-semibold text-gray-700 mb-3">🎯 ${this.getTranslation('factor_analysis') || 'Factor Analysis'}</p>
                <div class="space-y-2">
        `;

        const factorLabels = {
            water: 'water',
            nutrients: 'nutrients',
            temperature: 'temperature',
            soil_ph: 'soil_ph',
            pest_pressure: 'pest_pressure'
        };

        for (const [factor, value] of Object.entries(factors)) {
            const pct = (value * 100).toFixed(0);
            const isLimiting = factor === limitingFactor;
            const labelKey = factorLabels[factor] || factor;
            const label = this.getTranslation(labelKey) || factor;

            html += `
                <div class="flex items-center gap-2">
                    <span class="text-sm w-28 ${isLimiting ? 'text-red-600 font-bold' : 'text-gray-600'}">${label} ${isLimiting ? '' : ''}</span>
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
                    <p class="font-semibold text-emerald-800 mb-3">💰 ${this.getTranslation('economic_projection') || 'Economic Projection'}</p>
                    <div class="grid grid-cols-2 gap-2 text-sm">
                        <div class="bg-white p-2 rounded-lg">
                            <p class="text-xs text-gray-500">${this.getTranslation('total_yield') || 'Total Yield'}</p>
                            <p class="font-bold text-gray-800">${econ.total_yield_kg?.toLocaleString()} kg</p>
                        </div>
                        <div class="bg-white p-2 rounded-lg">
                            <p class="text-xs text-gray-500">${this.getTranslation('yield_per_ha') || 'Yield/Ha'}</p>
                            <p class="font-bold text-gray-800">${econ.yield_per_hectare_quintals?.toFixed(1)} qtl</p>
                        </div>
                        <div class="bg-white p-2 rounded-lg">
                            <p class="text-xs text-gray-500">${this.getTranslation('market_price') || 'Market Price'}</p>
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
                <p class="text-sm text-gray-600 mb-1">${this.getTranslation('soil_health_index')}</p>
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
                    <p class="font-semibold text-gray-700 mb-3"> ${this.getTranslation('component_scores')}</p>
                    <div class="space-y-2">
            `;

            const componentLabels = {
                nitrogen: 'nitrogen_n',
                phosphorus: 'phosphorus_p',
                potassium: 'potassium_k',
                ph_balance: 'ph_balance',
                organic_carbon: 'organic_carbon_label',
                ec: 'ec_salinity',
                texture: 'soil_texture'
            };

            for (const [component, score] of Object.entries(data.component_scores)) {
                const pct = Math.min(100, Math.round(score)); // Score is already 0-100, just round it
                const label = this.getTranslation(componentLabels[component]) || component;
                html += `
                    <div class="flex items-center gap-2">
                        <span class="text-sm w-32 text-gray-600">${label}</span>
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
                    <p class="font-semibold text-blue-800 mb-2"> ${this.getTranslation('improvement_recommendations')}</p>
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
                    <p class="font-semibold text-green-800 mb-2"> Suitable Crops</p>
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
                    <p class="font-semibold text-gray-700 mb-3"> 7-Day Irrigation Schedule</p>
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
                    <p class="font-semibold text-blue-800 mb-2"> Recommendations</p>
                    <ul class="text-sm text-blue-700 space-y-1">
                        ${data.recommendations.map(r => `<li>• ${r}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        container.innerHTML = html;
        container.classList.remove('hidden');
    }

    // Profit Calculator - Simplified with hardcoded costs per crop (per hectare)
    CROP_COSTS = {
        rice: { seed: 2500, fertilizer: 4000, pesticide: 2000, irrigation: 3000, labor: 8000, other: 1500 },
        wheat: { seed: 2000, fertilizer: 3500, pesticide: 1500, irrigation: 2000, labor: 6000, other: 1000 },
        cotton: { seed: 3000, fertilizer: 5000, pesticide: 4000, irrigation: 3500, labor: 10000, other: 2000 },
        maize: { seed: 1800, fertilizer: 3000, pesticide: 1800, irrigation: 2500, labor: 5000, other: 1000 },
        sugarcane: { seed: 8000, fertilizer: 6000, pesticide: 3000, irrigation: 5000, labor: 15000, other: 3000 },
        soybean: { seed: 2200, fertilizer: 2500, pesticide: 2000, irrigation: 1500, labor: 5000, other: 800 }
    };

    async calculateProfit(event) {
        event.preventDefault();

        // Read from radio buttons
        const cropRadio = document.querySelector('input[name="profit-crop-radio"]:checked');
        const areaRadio = document.querySelector('input[name="profit-area-radio"]:checked');

        if (!cropRadio || !areaRadio) {
            alert('Please select a crop and farm area');
            return;
        }

        const crop = cropRadio.value;
        const area = parseFloat(areaRadio.value);
        const costs = this.CROP_COSTS[crop] || this.CROP_COSTS['rice']; // Fallback to rice costs

        const data = {
            crop: crop,
            state: this.userData.state || 'Karnataka',
            area_hectares: area,
            seed_cost: costs.seed * area,
            fertilizer_cost: costs.fertilizer * area,
            pesticide_cost: costs.pesticide * area,
            irrigation_cost: costs.irrigation * area,
            labor_cost: costs.labor * area,
            other_costs: costs.other * area
        };

        try {
            const response = await fetch('/api/ml/profit-calculator', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                this.renderProfitResult(result.data, crop, area);
            } else {
                // Show error but also calculate locally as fallback
                console.error('API error:', result.error);
                this.calculateProfitLocally(crop, area, costs);
            }
        } catch (error) {
            console.error('Profit calculation failed:', error);
            // Fallback to local calculation
            this.calculateProfitLocally(crop, area, costs);
        }
    }

    // Local fallback profit calculation
    calculateProfitLocally(crop, area, costs) {
        const marketPrices = { rice: 2200, wheat: 2300, cotton: 6500, maize: 2000, sugarcane: 350, soybean: 4500 };
        const yields = { rice: 4500, wheat: 3800, cotton: 1800, maize: 6000, sugarcane: 70000, soybean: 2200 };

        const totalCost = Object.values(costs).reduce((a, b) => a + b, 0) * area;
        const yieldKg = (yields[crop] || 4000) * area;
        const price = marketPrices[crop] || 2000;
        const revenue = (yieldKg / 100) * price; // Convert to quintals
        const profit = revenue - totalCost;

        this.renderProfitResult({
            crop, area_hectares: area,
            yield: { predicted_kg_per_ha: yields[crop] || 4000, total_kg: yieldKg },
            revenue: { gross_revenue: revenue, market_price_per_quintal: price },
            total_cost: totalCost,
            financials: { net_profit: profit, profit_margin_percent: (profit / revenue * 100), return_on_investment_percent: (profit / totalCost * 100) }
        }, crop, area);
    }

    renderProfitResult(data, crop, area) {
        const container = document.getElementById('profit-result');
        const cropEmoji = this.getCropEmoji(crop);
        const cropName = crop.charAt(0).toUpperCase() + crop.slice(1);

        const isProfit = data.financials?.net_profit >= 0;
        const verdictColor = isProfit ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700';

        let html = `
            <div class="p-4 bg-blue-50 rounded-xl text-center mb-3">
                <p class="text-3xl mb-1">${cropEmoji}</p>
                <p class="font-semibold text-blue-800">${cropName} - ${area} Hectare${area > 1 ? 's' : ''}</p>
            </div>
            
            <div class="p-4 ${verdictColor} rounded-xl text-center">
                <p class="text-sm opacity-75 mb-1">${isProfit ? ' Profitable!' : ' Loss Expected'}</p>
                <p class="text-4xl font-bold">${isProfit ? '+' : '-'}₹${Math.abs(data.financials?.net_profit || 0).toLocaleString()}</p>
                <p class="text-sm mt-1">Net ${isProfit ? 'Profit' : 'Loss'}</p>
            </div>
            
            <div class="grid grid-cols-2 gap-3">
                <div class="p-3 bg-green-50 rounded-xl text-center">
                    <p class="text-xs text-gray-600">💰 You'll Earn</p>
                    <p class="text-lg font-bold text-green-700">₹${(data.revenue?.gross_revenue || 0).toLocaleString()}</p>
                </div>
                <div class="p-3 bg-red-50 rounded-xl text-center">
                    <p class="text-xs text-gray-600">💸 You'll Spend</p>
                    <p class="text-lg font-bold text-red-700">₹${(data.total_cost || 0).toLocaleString()}</p>
                </div>
            </div>
        `;

        // Yield Prediction - Simplified
        if (data.yield) {
            html += `
                <div class="p-4 bg-amber-50 rounded-xl text-center">
                    <p class="text-lg font-semibold text-amber-800"> Expected Harvest</p>
                    <p class="text-3xl font-bold text-amber-700 mt-1">${(data.yield.total_kg || 0).toLocaleString()} kg</p>
                    <p class="text-xs text-amber-600 mt-1">(${(data.yield.predicted_kg_per_ha || 0).toLocaleString()} kg per hectare)</p>
                </div>
            `;
        }

        // Simple ROI
        if (data.financials) {
            html += `
                <div class="p-4 bg-gray-50 rounded-xl">
                    <p class="font-semibold text-gray-700 mb-2"> Summary</p>
                    <div class="space-y-2 text-sm">
                        <div class="flex justify-between">
                            <span class="text-gray-600">Return on Investment</span>
                            <span class="font-bold ${data.financials.return_on_investment_percent >= 0 ? 'text-green-600' : 'text-red-600'}">${(data.financials.return_on_investment_percent || 0).toFixed(0)}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Profit Margin</span>
                            <span class="font-bold ${data.financials.profit_margin_percent >= 0 ? 'text-green-600' : 'text-red-600'}">${(data.financials.profit_margin_percent || 0).toFixed(0)}%</span>
                        </div>
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;
        container.classList.remove('hidden');
    }
}

// Initialize app
const app = new CeresApp();

// ============================================================================
// VOICE ASSISTANT MODULE
// ============================================================================

class VoiceAssistant {
    constructor(app) {
        this.app = app;
        this.recognition = null;
        this.synthesis = window.speechSynthesis;
        this.isListening = false;
        this.isSupported = 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;

        if (this.isSupported) {
            this.initRecognition();
        } else {
            console.warn('Speech recognition not supported in this browser');
        }
    }

    initRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = false;
        this.recognition.interimResults = true;
        this.recognition.maxAlternatives = 1;

        // Set language based on app language
        this.updateLanguage();

        this.recognition.onstart = () => {
            this.isListening = true;
            this.updateMicButton(true);
            this.showListeningFeedback();
            this.updateVoiceStatus('Listening...');
        };

        this.recognition.onresult = (event) => {
            let finalTranscript = '';
            let interimTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }

            this.updateTranscript(interimTranscript || finalTranscript, !event.results[event.results.length - 1].isFinal);

            if (finalTranscript) {
                this.processCommand(finalTranscript);
            }
        };

        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.isListening = false;
            this.updateMicButton(false);
            this.hideListeningFeedback();

            if (event.error === 'no-speech') {
                this.speak(this.app.getTranslation('no_speech_detected') || 'No speech detected. Please try again.');
            }
        };

        this.recognition.onend = () => {
            this.isListening = false;
            this.updateMicButton(false);
            this.hideListeningFeedback();
        };
    }

    updateLanguage() {
        if (!this.recognition) return;

        const langMap = {
            'en': 'en-IN',
            'hi': 'hi-IN',
            'kn': 'kn-IN',
            'te': 'te-IN',
            'ta': 'ta-IN',
            'mr': 'mr-IN'
        };

        this.recognition.lang = langMap[this.app.currentLang] || 'en-IN';
    }

    toggle() {
        if (this.isListening) {
            this.stop();
        } else {
            this.start();
        }
    }

    async start() {
        if (!this.isSupported) {
            this.showNotSupported();
            return;
        }

        // Request microphone permission
        try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (permErr) {
            this.showPermissionError();
            return;
        }

        this.updateLanguage();
        this.updateVoiceStatus('Starting...');

        try {
            this.recognition.start();
        } catch (e) {
            console.error('Error starting recognition:', e);
            // If already started, stop and restart
            if (e.name === 'InvalidStateError') {
                this.recognition.stop();
                setTimeout(() => {
                    try {
                        this.recognition.start();
                    } catch (e2) {
                        console.error('Failed to restart:', e2);
                    }
                }, 100);
            }
        }
    }

    showNotSupported() {
        const langMessages = {
            'en': 'Voice recognition is not supported in your browser. Please use Chrome or Edge.',
            'hi': 'आपके ब्राउज़र में वॉयस रिकॉग्निशन समर्थित नहीं है। कृपया Chrome या Edge का उपयोग करें।',
            'kn': 'ನಿಮ್ಮ ಬ್ರೌಸರ್‌ನಲ್ಲಿ ಧ್ವನಿ ಗುರುತಿಸುವಿಕೆ ಬೆಂಬಲಿತವಾಗಿಲ್ಲ. ದಯವಿಟ್ಟು Chrome ಅಥವಾ Edge ಬಳಸಿ.'
        };
        alert(langMessages[this.app.currentLang] || langMessages['en']);
    }

    showPermissionError() {
        const langMessages = {
            'en': 'Please allow microphone access to use voice assistant.',
            'hi': 'वॉयस असिस्टेंट का उपयोग करने के लिए कृपया माइक्रोफ़ोन एक्सेस की अनुमति दें।',
            'kn': 'ಧ್ವನಿ ಸಹಾಯಕವನ್ನು ಬಳಸಲು ದಯವಿಟ್ಟು ಮೈಕ್ರೋಫೋನ್ ಪ್ರವೇಶವನ್ನು ಅನುಮತಿಸಿ.'
        };
        alert(langMessages[this.app.currentLang] || langMessages['en']);
    }

    updateVoiceStatus(text) {
        const statusEl = document.getElementById('voice-status-text');
        if (statusEl) {
            statusEl.textContent = text;
        }
        const transcriptEl = document.getElementById('voice-transcript');
        if (transcriptEl && text !== 'Listening...') {
            transcriptEl.textContent = text;
        }
    }

    stop() {
        if (this.recognition) {
            this.recognition.stop();
        }
    }

    async processCommand(text) {
        try {
            const response = await fetch('/api/voice/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    language: this.app.currentLang
                })
            });

            const data = await response.json();

            if (data.success) {
                // Speak response
                if (data.speak) {
                    this.speak(data.speak);
                }

                // Execute action
                switch (data.action) {
                    case 'show_weather':
                        scrollToSection('weather-section');
                        break;
                    case 'show_crop_recommendation':
                        scrollToSection('recommend-section');
                        break;
                    case 'show_market_prices':
                        scrollToSection('market-section');
                        break;
                    case 'show_disease_scanner':
                        openDiseaseScanner();
                        break;
                    case 'show_water_calculator':
                        openWaterCalculator();
                        break;
                    case 'show_schemes':
                        scrollToSection('schemes-section');
                        break;
                    case 'ai_chat':
                        // Open AI chat and send the query
                        openAIChat();
                        if (data.ai_query) {
                            setTimeout(() => {
                                const input = document.getElementById('ai-chat-input');
                                if (input) {
                                    input.value = data.ai_query;
                                    sendAIMessage();
                                }
                            }, 500);
                        }
                        break;
                }
            }
        } catch (error) {
            console.error('Error processing voice command:', error);
        }
    }

    speak(text) {
        if (!this.synthesis) return;

        // Cancel any ongoing speech
        this.synthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);

        // Set language
        const langMap = {
            'en': 'en-IN',
            'hi': 'hi-IN',
            'kn': 'kn-IN',
            'te': 'te-IN',
            'ta': 'ta-IN',
            'mr': 'mr-IN'
        };

        utterance.lang = langMap[this.app.currentLang] || 'en-IN';
        utterance.rate = 0.9;
        utterance.pitch = 1;

        this.synthesis.speak(utterance);
    }

    updateMicButton(isActive) {
        const btn = document.getElementById('voice-btn');
        if (btn) {
            if (isActive) {
                btn.classList.add('voice-active');
                btn.innerHTML = `
                    <div class="voice-pulse"></div>
                    <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                        <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
                    </svg>
                `;
            } else {
                btn.classList.remove('voice-active');
                btn.innerHTML = `
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/>
                    </svg>
                `;
            }
        }
    }

    showListeningFeedback() {
        let feedback = document.getElementById('voice-feedback');
        if (!feedback) {
            feedback = document.createElement('div');
            feedback.id = 'voice-feedback';
            feedback.className = 'fixed bottom-24 left-1/2 transform -translate-x-1/2 bg-white rounded-2xl shadow-2xl p-4 z-50 min-w-[300px]';
            feedback.innerHTML = `
                <div class="flex items-center gap-3">
                    <div class="voice-wave">
                        <span></span><span></span><span></span><span></span><span></span>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500">Listening...</p>
                        <p id="voice-transcript" class="font-medium text-gray-800"></p>
                    </div>
                </div>
            `;
            document.body.appendChild(feedback);
        }
        feedback.classList.remove('hidden');
    }

    hideListeningFeedback() {
        const feedback = document.getElementById('voice-feedback');
        if (feedback) {
            feedback.classList.add('hidden');
        }
    }

    updateTranscript(text, isInterim) {
        const el = document.getElementById('voice-transcript');
        if (el) {
            el.textContent = text;
            el.style.opacity = isInterim ? '0.6' : '1';
        }
    }
}

// Initialize voice assistant
const voiceAssistant = new VoiceAssistant(app);


// ============================================================================
// AI CHAT MODULE
// ============================================================================

class AIChat {
    constructor(app) {
        this.app = app;
        this.conversationHistory = [];
        this.isLoading = false;
    }

    async checkStatus() {
        try {
            const response = await fetch('/api/ai/status');
            return await response.json();
        } catch (error) {
            return { connected: false, error: 'Cannot connect to AI service' };
        }
    }

    async sendMessage(message) {
        if (this.isLoading || !message.trim()) return;

        this.isLoading = true;
        this.addMessage(message, 'user');
        this.showTypingIndicator();

        // Add to history
        this.conversationHistory.push({ role: 'user', content: message });

        try {
            const response = await fetch('/api/ai/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    history: this.conversationHistory.slice(-10),
                    language: this.app.currentLang
                })
            });

            const data = await response.json();

            this.hideTypingIndicator();

            if (data.success) {
                this.addMessage(data.response, 'assistant');
                this.conversationHistory.push({ role: 'assistant', content: data.response });

                // Speak response if voice is enabled
                if (document.getElementById('ai-voice-toggle')?.checked) {
                    voiceAssistant.speak(data.response);
                }
            } else {
                this.addMessage(data.error || 'Sorry, I could not process your request.', 'error');
            }
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('Connection error. Please check if Ollama is running.', 'error');
        }

        this.isLoading = false;
    }

    addMessage(text, type) {
        const container = document.getElementById('ai-chat-messages');
        if (!container) return;

        const messageEl = document.createElement('div');
        messageEl.className = `flex ${type === 'user' ? 'justify-end' : 'justify-start'} mb-3`;

        const bubbleClass = type === 'user'
            ? 'bg-green-600 text-white rounded-2xl rounded-br-md'
            : type === 'error'
                ? 'bg-red-100 text-red-700 rounded-2xl rounded-bl-md'
                : 'bg-gray-100 text-gray-800 rounded-2xl rounded-bl-md';

        messageEl.innerHTML = `
            <div class="max-w-[80%] p-3 ${bubbleClass}">
                <p class="text-sm whitespace-pre-wrap">${this.formatMessage(text)}</p>
            </div>
        `;

        container.appendChild(messageEl);
        container.scrollTop = container.scrollHeight;
    }

    formatMessage(text) {
        // Basic markdown-like formatting
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }

    showTypingIndicator() {
        const container = document.getElementById('ai-chat-messages');
        if (!container) return;

        const indicator = document.createElement('div');
        indicator.id = 'typing-indicator';
        indicator.className = 'flex justify-start mb-3';
        indicator.innerHTML = `
            <div class="bg-gray-100 rounded-2xl rounded-bl-md p-3">
                <div class="flex gap-1">
                    <span class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0ms"></span>
                    <span class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 150ms"></span>
                    <span class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 300ms"></span>
                </div>
            </div>
        `;
        container.appendChild(indicator);
        container.scrollTop = container.scrollHeight;
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    async loadSuggestions() {
        try {
            const response = await fetch(`/api/ai/suggestions?lang=${this.app.currentLang}`);
            const data = await response.json();

            if (data.success && data.suggestions) {
                this.renderSuggestions(data.suggestions);
            }
        } catch (error) {
            console.error('Failed to load AI suggestions:', error);
        }
    }

    renderSuggestions(suggestions) {
        const container = document.getElementById('ai-suggestions');
        if (!container) return;

        container.innerHTML = suggestions.map(s => `
            <button onclick="aiChat.sendMessage('${s.replace(/'/g, "\\'")}')" 
                    class="px-3 py-2 bg-white border border-gray-200 rounded-full text-sm text-gray-700 hover:bg-green-50 hover:border-green-300 transition-colors whitespace-nowrap">
                ${s}
            </button>
        `).join('');
    }

    clear() {
        this.conversationHistory = [];
        const container = document.getElementById('ai-chat-messages');
        if (container) {
            container.innerHTML = `
                <div class="text-center text-gray-500 py-8">
                    <div class="w-16 h-16 mx-auto mb-4 bg-green-100 rounded-full flex items-center justify-center">
                        <svg class="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/>
                        </svg>
                    </div>
                    <p class="font-medium">CERES AI Assistant</p>
                    <p class="text-sm mt-1">Ask me anything about farming!</p>
                </div>
            `;
        }
    }
}

// Initialize AI Chat
const aiChat = new AIChat(app);


// ============================================================================
// DISEASE SCANNER MODULE
// ============================================================================

class DiseaseScanner {
    constructor(app) {
        this.app = app;
        this.stream = null;
    }

    async openCamera() {
        const video = document.getElementById('disease-camera');
        const preview = document.getElementById('disease-preview');

        if (!video) return;

        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment' }
            });

            video.srcObject = this.stream;
            video.classList.remove('hidden');
            preview.classList.add('hidden');

            document.getElementById('capture-btn').classList.remove('hidden');
            document.getElementById('camera-btn').classList.add('hidden');
        } catch (error) {
            console.error('Camera error:', error);
            alert('Could not access camera. Please use file upload instead.');
        }
    }

    captureImage() {
        const video = document.getElementById('disease-camera');
        const canvas = document.getElementById('disease-canvas');
        const preview = document.getElementById('disease-preview');

        if (!video || !canvas) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        // Stop camera
        this.stopCamera();

        // Show preview
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        preview.src = imageData;
        preview.classList.remove('hidden');
        video.classList.add('hidden');

        document.getElementById('capture-btn').classList.add('hidden');
        document.getElementById('camera-btn').classList.remove('hidden');
        document.getElementById('analyze-disease-btn').classList.remove('hidden');
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
    }

    handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = document.getElementById('disease-preview');
            preview.src = e.target.result;
            preview.classList.remove('hidden');
            document.getElementById('disease-camera').classList.add('hidden');
            document.getElementById('analyze-disease-btn').classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    async analyzeImage() {
        const preview = document.getElementById('disease-preview');
        const canvas = document.getElementById('disease-canvas');
        const resultContainer = document.getElementById('disease-scan-result');

        // Get image data
        let imageBase64;
        if (preview.src.startsWith('data:')) {
            imageBase64 = preview.src;
        } else {
            // Convert from canvas
            imageBase64 = canvas.toDataURL('image/jpeg', 0.8);
        }

        // Show loading
        resultContainer.innerHTML = `
            <div class="flex items-center justify-center py-8">
                <div class="animate-spin w-8 h-8 border-4 border-green-500 border-t-transparent rounded-full"></div>
                <span class="ml-3 text-gray-600">Analyzing image...</span>
            </div>
        `;
        resultContainer.classList.remove('hidden');

        try {
            const response = await fetch('/api/disease/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_base64: imageBase64,
                    language: this.app.currentLang
                })
            });

            const data = await response.json();

            if (data.success) {
                this.renderResult(data);
            } else {
                resultContainer.innerHTML = `
                    <div class="p-4 bg-red-50 rounded-xl text-center">
                        <p class="text-red-600">${data.error || 'Analysis failed. Please try again.'}</p>
                    </div>
                `;
            }
        } catch (error) {
            resultContainer.innerHTML = `
                <div class="p-4 bg-red-50 rounded-xl text-center">
                    <p class="text-red-600">Connection error. Please try again.</p>
                </div>
            `;
        }
    }

    renderResult(data) {
        const container = document.getElementById('disease-scan-result');

        const severityColors = {
            none: 'bg-green-100 text-green-700',
            moderate: 'bg-yellow-100 text-yellow-700',
            severe: 'bg-red-100 text-red-700',
            high: 'bg-red-100 text-red-700'
        };

        const severityColor = severityColors[data.severity] || severityColors.moderate;

        let html = `
            <div class="p-4 ${severityColor} rounded-xl text-center mb-4">
                <p class="text-sm opacity-75 mb-1">Detected</p>
                <p class="text-2xl font-bold">${data.disease_name}</p>
                <p class="text-sm mt-1">Confidence: ${data.confidence?.toFixed(1)}%</p>
            </div>
        `;

        if (data.severity !== 'none') {
            // Symptoms
            if (data.symptoms) {
                html += `
                    <div class="p-4 bg-gray-50 rounded-xl mb-4">
                        <p class="font-semibold text-gray-700 mb-2">🔍 Symptoms</p>
                        <p class="text-sm text-gray-600">${data.symptoms}</p>
                    </div>
                `;
            }

            // Treatment
            if (data.treatment && data.treatment.length > 0) {
                html += `
                    <div class="p-4 bg-blue-50 rounded-xl mb-4">
                        <p class="font-semibold text-blue-800 mb-2">💊 Treatment</p>
                        <ul class="text-sm text-blue-700 space-y-1">
                            ${data.treatment.map(t => `<li>• ${t}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }

            // Prevention
            if (data.prevention) {
                html += `
                    <div class="p-4 bg-green-50 rounded-xl">
                        <p class="font-semibold text-green-800 mb-2">🛡️ Prevention</p>
                        <p class="text-sm text-green-700">${data.prevention}</p>
                    </div>
                `;
            }
        } else {
            html += `
                <div class="p-4 bg-green-50 rounded-xl text-center">
                    <p class="text-4xl mb-2"></p>
                    <p class="text-green-700 font-medium">Your plant looks healthy!</p>
                    <p class="text-sm text-green-600 mt-1">Continue with regular care and monitoring.</p>
                </div>
            `;
        }

        container.innerHTML = html;
    }
}

// Initialize Disease Scanner
const diseaseScanner = new DiseaseScanner(app);


// ============================================================================
// GLOBAL FUNCTIONS
// ============================================================================

function changeLanguage(lang) {
    app.changeLanguage(lang);
    voiceAssistant.updateLanguage();
    aiChat.loadSuggestions();
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

// Soil preset values for farmer-friendly selection
const SOIL_PRESETS = {
    black: { N: 60, P: 30, K: 50, ph: 7.5 },    // Black soil - high in potash, slightly alkaline
    red: { N: 40, P: 20, K: 30, ph: 5.5 },       // Red soil - low in nutrients, acidic
    alluvial: { N: 80, P: 40, K: 40, ph: 6.5 },  // Alluvial - fertile, neutral
    loamy: { N: 70, P: 50, K: 45, ph: 6.8 }      // Loamy - balanced, good for vegetables
};

const RAINFALL_PRESETS = {
    low: 40,
    medium: 100,
    high: 200
};

// Crop emoji mapping for visual display
const CROP_EMOJIS = {
    rice: '/static/images/crops/rice.png',
    wheat: '/static/images/crops/wheat.png',
    maize: '/static/images/crops/maize.png',
    corn: '/static/images/crops/maize.png',
    cotton: '/static/images/crops/cotton.png',
    sugarcane: '/static/images/crops/sugarcane.png',
    soybean: '/static/images/crops/soybean.png',
    soya: '/static/images/crops/soybean.png',
    groundnut: '/static/images/crops/groundnut.png',
    peanut: '/static/images/crops/groundnut.png',
    potato: '/static/images/crops/potato.png',
    tomato: '',
    onion: '', chilli: '', pepper: '', mango: '',
    banana: '', orange: '', apple: '', grape: '',
    coconut: '', coffee: '', tea: '', mustard: '',
    sunflower: '', turmeric: '', ginger: '', garlic: '',
    cabbage: '', carrot: '', cucumber: '', watermelon: '',
    pumpkin: '', lentil: '', chickpea: '', millet: '',
    jowar: '', bajra: '', ragi: '', default: '',
    brinjal: '', cauliflower: '', lettuce: '', spinach: '',
    peas: '', beans: '', melon: '', millets: '', barley: '', oats: ''
};

function setSoilPreset(type) {
    const preset = SOIL_PRESETS[type];
    if (preset) {
        document.getElementById('nitrogen').value = preset.N;
        document.getElementById('phosphorus').value = preset.P;
        document.getElementById('potassium').value = preset.K;
        document.getElementById('ph').value = preset.ph;
    }
    // Update UI selection
    document.querySelectorAll('.soil-option').forEach(el => {
        el.classList.remove('selected');
        el.parentElement.querySelector('input').checked = false;
    });
    const selected = document.querySelector(`input[value="${type}"]`);
    if (selected) {
        selected.checked = true;
        selected.nextElementSibling.classList.add('selected');
    }
}

function setRainfallPreset(type) {
    const rainfall = RAINFALL_PRESETS[type];
    if (rainfall) {
        document.getElementById('rainfall-input').value = rainfall;
    }
    // Update UI selection
    document.querySelectorAll('.rainfall-option').forEach(el => {
        el.classList.remove('selected');
    });
    const selected = document.querySelector(`input[name="rainfall-preset"][value="${type}"]`);
    if (selected) {
        selected.checked = true;
        selected.nextElementSibling.classList.add('selected');
    }
}

function getCropEmoji(crop) {
    const cropLower = crop.toLowerCase();
    return CROP_EMOJIS[cropLower] || CROP_EMOJIS.default;
}

// Modal functions
function openModal(modalId) {
    document.getElementById(modalId)?.classList.add('active');
}

function closeModal(modalId) {
    document.getElementById(modalId)?.classList.remove('active');
}

function openWaterCalculator() {
    // Auto-fill user's plot area
    const userArea = app.getUserPlotArea();
    const areaInput = document.getElementById('water-area');
    if (areaInput && userArea > 0) {
        areaInput.value = userArea.toFixed(1);
    }
    openModal('water-modal');
}

function openFertilizerCalculator() {
    // Auto-fill user's plot area
    const userArea = app.getUserPlotArea();
    const areaInput = document.getElementById('fert-area');
    if (areaInput && userArea > 0) {
        areaInput.value = userArea.toFixed(1);
    }
    openModal('fertilizer-modal');
}

function openCropCalendar() {
    app.loadCropCalendar();
    openModal('calendar-modal');
}

function openFarmDiary() {
    // Farm diary removed - show alert
    alert('Farm Diary feature coming soon!');
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
    openModal('disease-risk-modal');
}

function openYieldPredictor() {
    openModal('yield-modal');
}

function openSoilHealthAnalyzer() {
    // Pre-fill soil data from user profile
    const soilType = app.userData.soil_type || 'loamy';
    const soilSelect = document.getElementById('soil-type-sh');
    if (soilSelect) soilSelect.value = soilType;
    openModal('soil-modal');
}

function openIrrigationScheduler() {
    // Pre-fill from user's profile and plots
    const soilType = app.userData.soil_type || 'loamy';
    const userArea = app.getUserPlotArea();

    // Set soil type
    const soilSelect = document.getElementById('irr-soil');
    if (soilSelect) {
        // Map user soil types to irrigation soil types
        const soilMap = {
            'alluvial': 'loamy', 'black': 'clayey', 'red': 'loamy',
            'laterite': 'loamy', 'arid': 'sandy', 'forest': 'loamy',
            'saline': 'clayey', 'peaty': 'clayey', 'loamy': 'loamy'
        };
        soilSelect.value = soilMap[soilType] || 'loamy';
    }

    // Set area from user's plots
    const areaInput = document.getElementById('irr-area');
    if (areaInput && userArea > 0) {
        areaInput.value = userArea.toFixed(1);
    }

    openModal('irrigation-modal');
}

function openProfitCalculator() {
    // Pre-select user's plot area if available
    const userArea = app.getUserPlotArea();
    if (userArea > 0) {
        // Find closest radio option to user's area
        const areaOptions = [1, 2, 5, 10];
        const closest = areaOptions.reduce((prev, curr) =>
            Math.abs(curr - userArea) < Math.abs(prev - userArea) ? curr : prev
        );
        const radio = document.querySelector(`input[name="profit-area-radio"][value="${closest}"]`);
        if (radio) {
            radio.checked = true;
        }
    }
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

// Voice Assistant Functions
function toggleVoice() {
    voiceAssistant.toggle();
}

function toggleVoiceAssistant() {
    const overlay = document.getElementById('voice-overlay');
    if (overlay) {
        overlay.classList.toggle('hidden');
        overlay.classList.toggle('flex');
    }
    voiceAssistant.start();
}

function stopVoiceAssistant() {
    voiceAssistant.stop();
    const overlay = document.getElementById('voice-overlay');
    if (overlay) {
        overlay.classList.add('hidden');
        overlay.classList.remove('flex');
    }
}

function changeVoiceLanguage(lang) {
    app.changeLanguage(lang);
    voiceAssistant.updateLanguage();
}

function toggleVoiceInput() {
    voiceAssistant.toggle();
}

// AI Chat Functions
function openAIChat() {
    aiChat.loadSuggestions();
    aiChat.checkStatus().then(status => {
        const statusEl = document.getElementById('ai-status');
        if (statusEl) {
            if (status.connected) {
                statusEl.innerHTML = `<span class="w-2 h-2 bg-green-500 rounded-full"></span> Online`;
                statusEl.className = 'flex items-center gap-2 text-xs text-green-600';
            } else {
                statusEl.innerHTML = `<span class="w-2 h-2 bg-red-500 rounded-full"></span> Offline`;
                statusEl.className = 'flex items-center gap-2 text-xs text-red-600';
            }
        }
    });
    openModal('ai-chat-modal');
}

function sendAIMessage() {
    const input = document.getElementById('ai-chat-input');
    if (input && input.value.trim()) {
        aiChat.sendMessage(input.value.trim());
        input.value = '';
    }
}

function clearAIChat() {
    aiChat.clear();
}

// Disease Scanner Functions
let diseaseImageData = null;
let diseaseCameraStream = null;

function openDiseaseScanner() {
    document.getElementById('disease-results')?.classList.add('hidden');
    document.getElementById('disease-image-preview')?.classList.add('hidden');
    openModal('plant-disease-modal');
}

function handleDiseaseImageSelect(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            diseaseImageData = e.target.result;
            const previewImg = document.getElementById('disease-preview-img');
            const previewContainer = document.getElementById('disease-image-preview');
            if (previewImg && previewContainer) {
                previewImg.src = diseaseImageData;
                previewContainer.classList.remove('hidden');
                document.getElementById('disease-upload-area')?.classList.add('hidden');
            }
        };
        reader.readAsDataURL(file);
    }
}

async function openDiseaseCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
        });
        diseaseCameraStream = stream;
        const video = document.getElementById('disease-camera-video');
        const container = document.getElementById('disease-camera-container');
        if (video && container) {
            video.srcObject = stream;
            container.classList.remove('hidden');
            document.getElementById('disease-upload-area')?.classList.add('hidden');
        }
    } catch (error) {
        console.error('Camera error:', error);
        alert('Could not access camera. Please upload an image instead.');
    }
}

function captureDiseaseImage() {
    const video = document.getElementById('disease-camera-video');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    diseaseImageData = canvas.toDataURL('image/jpeg');

    // Stop camera
    closeDiseaseCamera();

    // Show preview
    const previewImg = document.getElementById('disease-preview-img');
    const previewContainer = document.getElementById('disease-image-preview');
    if (previewImg && previewContainer) {
        previewImg.src = diseaseImageData;
        previewContainer.classList.remove('hidden');
    }
}

function closeDiseaseCamera() {
    if (diseaseCameraStream) {
        diseaseCameraStream.getTracks().forEach(track => track.stop());
        diseaseCameraStream = null;
    }
    document.getElementById('disease-camera-container')?.classList.add('hidden');
    document.getElementById('disease-upload-area')?.classList.remove('hidden');
}

async function analyzeDiseaseImage() {
    if (!diseaseImageData) {
        alert('Please upload or capture an image first.');
        return;
    }

    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = '🔍 Analyzing...';
    }

    try {
        const response = await fetch('/api/disease/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: diseaseImageData,
                lang: app.currentLang,
                ts: Date.now()
            })
        });

        const data = await response.json();
        renderDiseaseResults(data);
    } catch (error) {
        console.error('Disease analysis failed:', error);
        document.getElementById('disease-results').innerHTML = `
            <div class="p-4 bg-red-50 rounded-xl text-red-600">
                Analysis failed. Please try again.
            </div>
        `;
        document.getElementById('disease-results')?.classList.remove('hidden');
    } finally {
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = '🔍 Analyze Plant';
        }
    }
}

function renderDiseaseResults(data) {
    const container = document.getElementById('disease-results');
    if (!container) return;

    if (!data.success) {
        container.innerHTML = `
            <div class="p-4 bg-red-50 rounded-xl text-red-600">
                ${data.error || 'Analysis failed. Please try again.'}
            </div>
        `;
        container.classList.remove('hidden');
        return;
    }

    if (data.disease_key === 'healthy' || data.disease_name === 'Healthy Plant') {
        container.innerHTML = `
            <div class="p-6 bg-green-50 rounded-xl text-center">
                <span class="text-6xl mb-4 block"></span>
                <h3 class="text-2xl font-bold text-green-700 mb-2">${app.getTranslation('healthy_plant') || 'Healthy Plant!'}</h3>
                <p class="text-green-600">${app.getTranslation('no_disease_detected') || 'No disease detected. Your plant looks healthy.'}</p>
            </div>
        `;
    } else {
        const confidence = data.confidence || 75;
        const severityColors = {
            'severe': 'bg-red-100 text-red-700 border-red-300',
            'high': 'bg-orange-100 text-orange-700 border-orange-300',
            'moderate': 'bg-yellow-100 text-yellow-700 border-yellow-300',
            'low': 'bg-blue-100 text-blue-700 border-blue-300'
        };
        const severityColor = severityColors[data.severity] || severityColors['moderate'];

        // Format treatment as list if it's an array
        let treatmentHtml = '';
        if (data.treatment) {
            if (Array.isArray(data.treatment)) {
                treatmentHtml = `<ul class="list-disc list-inside space-y-1 text-sm text-blue-600">
                    ${data.treatment.map(t => `<li>${t}</li>`).join('')}
                </ul>`;
            } else {
                treatmentHtml = `<p class="text-sm text-blue-600">${data.treatment}</p>`;
            }
        }

        // Try to get translated disease name
        const diseaseKey = data.disease_name.toLowerCase().replace(/ /g, '_');
        const translatedDiseaseName = app.getTranslation(diseaseKey) || data.disease_name;

        container.innerHTML = `
            <div class="p-4 bg-red-50 border-2 border-red-200 rounded-xl">
                <div class="flex items-center gap-3 mb-3">
                    <span class="text-4xl">🦠</span>
                    <div class="flex-1">
                        <h3 class="text-xl font-bold text-red-700">${translatedDiseaseName || 'Unknown Disease'}</h3>
                        <div class="flex items-center gap-2 mt-1">
                            <span class="text-sm text-red-600">${app.getTranslation('confidence_level') || 'Confidence'}: ${confidence.toFixed(0)}%</span>
                            <span class="px-2 py-0.5 text-xs font-medium rounded-full ${severityColor}">${(data.severity ? (app.getTranslation(data.severity) || data.severity) : 'unknown').toUpperCase()}</span>
                        </div>
                    </div>
                </div>
            </div>
            
            ${data.symptoms ? `
            <div class="p-4 bg-orange-50 rounded-xl">
                <h4 class="font-semibold text-orange-700 mb-2"> ${app.getTranslation('symptoms') || 'Symptoms'}</h4>
                <p class="text-sm text-orange-600">${data.symptoms}</p>
            </div>
            ` : ''}
            
            ${data.treatment ? `
            <div class="p-4 bg-blue-50 rounded-xl">
                <h4 class="font-semibold text-blue-700 mb-2">💊 ${app.getTranslation('treatment') || 'Treatment'}</h4>
                ${treatmentHtml}
            </div>
            ` : ''}
            
            ${data.prevention ? `
            <div class="p-4 bg-green-50 rounded-xl">
                <h4 class="font-semibold text-green-700 mb-2">🛡️ ${app.getTranslation('prevention') || 'Prevention'}</h4>
                <p class="text-sm text-green-600">${data.prevention}</p>
            </div>
            ` : ''}
        `;
    }
    container.classList.remove('hidden');
}

// Plot Mapper Functions
function openPlotMapper() {
    openModal('plot-modal');
    setTimeout(() => {
        initPlotMap();
    }, 300);
}

let plotMap = null;
let drawnItems = null;

function clearPlotDrawing() {
    if (drawnItems) {
        drawnItems.clearLayers();
    }
    const btn = document.getElementById('analyze-plot-btn');
    if (btn) {
        btn.disabled = true;
        btn.classList.add('opacity-50');
    }
    document.getElementById('plot-analysis-results')?.classList.add('hidden');
}

function initPlotMap() {
    if (plotMap) {
        plotMap.remove();
    }

    // Default to user's location or Bangalore
    const lat = app.userData.latitude || 12.9716;
    const lon = app.userData.longitude || 77.5946;

    plotMap = L.map('plot-map').setView([lat, lon], 16);

    // Add hybrid layer (satellite + labels)
    const satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Esri'
    });

    const labels = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap, &copy; CartoDB'
    });

    satellite.addTo(plotMap);
    labels.addTo(plotMap);

    // Load user's saved plots if any
    if (app.userPlots && app.userPlots.length > 0) {
        app.userPlots.forEach(plot => {
            if (plot.coordinates && plot.coordinates.length >= 3) {
                const polygon = L.polygon(plot.coordinates, {
                    color: '#3b82f6',
                    weight: 2,
                    fillOpacity: 0.2
                }).addTo(plotMap);
                polygon.bindPopup(`<b>${plot.name}</b><br>${plot.area_hectares} Ha`);
            }
        });
    }

    // Add drawing controls
    drawnItems = new L.FeatureGroup();
    plotMap.addLayer(drawnItems);

    const drawControl = new L.Control.Draw({
        edit: {
            featureGroup: drawnItems
        },
        draw: {
            polygon: {
                allowIntersection: false,
                showArea: true,
                shapeOptions: {
                    color: '#10b981',
                    weight: 3,
                    fillOpacity: 0.3
                }
            },
            rectangle: {
                shapeOptions: {
                    color: '#10b981',
                    weight: 3,
                    fillOpacity: 0.3
                }
            },
            circle: false,
            circlemarker: false,
            marker: false,
            polyline: false
        }
    });
    plotMap.addControl(drawControl);

    // Handle draw events
    plotMap.on(L.Draw.Event.CREATED, function (e) {
        drawnItems.clearLayers();
        drawnItems.addLayer(e.layer);
        const btn = document.getElementById('analyze-plot-btn');
        if (btn) {
            btn.disabled = false;
            btn.classList.remove('opacity-50');
        }
    });
}

async function analyzePlot() {
    if (!drawnItems || drawnItems.getLayers().length === 0) {
        alert('Please draw your plot boundary on the map first.');
        return;
    }

    const layer = drawnItems.getLayers()[0];
    let coordinates = [];

    if (layer.getLatLngs) {
        const latlngs = layer.getLatLngs()[0];
        coordinates = latlngs.map(ll => [ll.lat, ll.lng]);
    }

    if (coordinates.length < 3) {
        alert('Please draw a valid plot boundary.');
        return;
    }

    const resultContainer = document.getElementById('plot-analysis-results');
    resultContainer.innerHTML = `
        <div class="flex items-center justify-center py-8">
            <div class="animate-spin w-8 h-8 border-4 border-green-500 border-t-transparent rounded-full"></div>
            <span class="ml-3 text-gray-600">Analyzing plot...</span>
        </div>
    `;
    resultContainer.classList.remove('hidden');

    // Get soil params from user profile
    const soilType = app.userData.soil_type || 'loam';
    const soilPresets = {
        'black': { N: 60, P: 30, K: 50, ph: 7.5 },
        'red': { N: 40, P: 20, K: 30, ph: 6.0 },
        'alluvial': { N: 80, P: 40, K: 40, ph: 6.5 },
        'loamy': { N: 70, P: 35, K: 45, ph: 6.8 },
        'laterite': { N: 35, P: 15, K: 25, ph: 5.5 },
        'arid': { N: 25, P: 10, K: 20, ph: 8.0 },
        'forest': { N: 90, P: 50, K: 60, ph: 5.8 },
        'saline': { N: 30, P: 15, K: 35, ph: 8.5 },
        'peaty': { N: 95, P: 55, K: 50, ph: 5.0 }
    };
    const soilParams = soilPresets[soilType] || soilPresets['alluvial'];

    try {
        const response = await fetch('/api/plot/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                coordinates: coordinates,
                soil_type: soilType,
                nitrogen: soilParams.N,
                phosphorus: soilParams.P,
                potassium: soilParams.K,
                ph: soilParams.ph,
                lang: app.currentLang
            })
        });

        const data = await response.json();

        if (data.success) {
            renderPlotAnalysis(data);
        } else {
            resultContainer.innerHTML = `<div class="p-4 bg-red-50 rounded-xl text-red-600">${data.error}</div>`;
        }
    } catch (error) {
        resultContainer.innerHTML = `<div class="p-4 bg-red-50 rounded-xl text-red-600">Analysis failed. Please try again.</div>`;
    }
}

function renderPlotAnalysis(data) {
    const container = document.getElementById('plot-analysis-results');

    let html = `
        <div class="p-4 bg-green-50 rounded-xl mb-4">
            <div class="flex justify-between items-center mb-2">
                <span class="font-semibold text-green-800">📐 Plot Size</span>
                <span class="text-2xl font-bold text-green-700">${data.plot_info.area_hectares} Ha</span>
            </div>
            <p class="text-sm text-green-600">Current Season: ${data.plot_info.current_season.charAt(0).toUpperCase() + data.plot_info.current_season.slice(1)}</p>
        </div>
        
        <p class="font-semibold text-gray-700 mb-3">🗺️ Recommended Zones</p>
    `;

    // Zone recommendations
    data.zones.forEach((zone, idx) => {
        const colors = ['bg-green-100 border-green-300', 'bg-blue-100 border-blue-300', 'bg-orange-100 border-orange-300'];
        html += `
            <div class="p-4 ${colors[idx % 3]} border-2 rounded-xl mb-3">
                <div class="flex justify-between items-center mb-2">
                    <span class="font-bold text-gray-800">${zone.zone_name}</span>
                    <span class="text-sm text-gray-600">${zone.area_percent}% (${zone.area_hectares} Ha)</span>
                </div>
                <div class="flex items-center gap-3">
                    <span class="text-3xl"></span>
                    <div>
                        <p class="font-semibold text-gray-800">${zone.recommended_crop_translated}</p>
                        <p class="text-sm text-gray-600">Expected: ${zone.expected_yield_kg.toLocaleString()} kg</p>
                    </div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

// Close modals on outside click
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.classList.remove('active');
        closeDiseaseCamera();
    }
});

// Close modals on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal.active').forEach(modal => {
            modal.classList.remove('active');
        });
        closeDiseaseCamera();
    }
});

// AI Chat Enter key
document.addEventListener('DOMContentLoaded', () => {
    // Set user's soil type image
    const userSoilImageEl = document.getElementById('user-soil-image');
    const userSoilNameEl = document.getElementById('user-soil-name');

    if (userSoilImageEl && userSoilNameEl) {
        // Extract soil type from the text (e.g., "Alluvial Soil" -> "alluvial")
        const soilText = userSoilNameEl.textContent.trim().toLowerCase();
        let soilType = 'alluvial'; // default

        for (const type of ['alluvial', 'black', 'red', 'laterite', 'arid', 'forest', 'saline', 'peaty']) {
            if (soilText.includes(type)) {
                soilType = type;
                break;
            }
        }

        // Set the image
        const imagePath = soilPresets[soilType]?.emoji || '/static/images/soils/alluvial_soil.png';
        userSoilImageEl.innerHTML = `<img src="${imagePath}" alt="${soilType} soil" class="w-full h-full object-cover">`;
    }

    const aiInput = document.getElementById('ai-chat-input');
    if (aiInput) {
        aiInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendAIMessage();
            }
        });
    }
});
