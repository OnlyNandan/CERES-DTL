from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    language = db.Column(db.String(5), default='en')
    state = db.Column(db.String(100))
    district = db.Column(db.String(100))
    city = db.Column(db.String(100))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    soil_type = db.Column(db.String(50))
    farm_size = db.Column(db.Float)  # in hectares
    
    setup_complete = db.Column(db.Boolean, default=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'language': self.language,
            'state': self.state,
            'district': self.district,
            'city': self.city,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'soil_type': self.soil_type,
            'farm_size': self.farm_size,
            'setup_complete': self.setup_complete
        }


class SoilData(db.Model):
    __tablename__ = 'soil_data'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    nitrogen = db.Column(db.Float)
    phosphorus = db.Column(db.Float)
    potassium = db.Column(db.Float)
    ph = db.Column(db.Float)
    rainfall = db.Column(db.Float)
    temperature = db.Column(db.Float)
    humidity = db.Column(db.Float)
    
    user = db.relationship('User', backref=db.backref('soil_records', lazy=True))


class CropRecommendation(db.Model):
    __tablename__ = 'crop_recommendations'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    recommended_crop = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    
    nitrogen = db.Column(db.Float)
    phosphorus = db.Column(db.Float)
    potassium = db.Column(db.Float)
    ph = db.Column(db.Float)
    rainfall = db.Column(db.Float)
    
    user = db.relationship('User', backref=db.backref('recommendations', lazy=True))


class MarketPrice(db.Model):
    __tablename__ = 'market_prices'
    
    id = db.Column(db.Integer, primary_key=True)
    fetched_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    state = db.Column(db.String(100))
    district = db.Column(db.String(100))
    market = db.Column(db.String(100))
    commodity = db.Column(db.String(100))
    variety = db.Column(db.String(100))
    
    min_price = db.Column(db.Float)
    max_price = db.Column(db.Float)
    modal_price = db.Column(db.Float)
    
    arrival_date = db.Column(db.Date)


class FarmDiary(db.Model):
    __tablename__ = 'farm_diary'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    entry_date = db.Column(db.Date, default=datetime.utcnow)
    activity_type = db.Column(db.String(50))  # sowing, irrigation, fertilizer, harvest, etc.
    crop = db.Column(db.String(100))
    notes = db.Column(db.Text)
    expense = db.Column(db.Float, default=0)
    income = db.Column(db.Float, default=0)
    
    user = db.relationship('User', backref=db.backref('diary_entries', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'activity_type': self.activity_type,
            'crop': self.crop,
            'notes': self.notes,
            'expense': self.expense,
            'income': self.income,
            'created_at': self.created_at.isoformat()
        }


class WeatherAlert(db.Model):
    __tablename__ = 'weather_alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    alert_type = db.Column(db.String(50))  # rain, frost, heatwave, pest
    severity = db.Column(db.String(20))  # low, medium, high
    message = db.Column(db.Text)
    is_read = db.Column(db.Boolean, default=False)
    
    user = db.relationship('User', backref=db.backref('alerts', lazy=True))


class CropPlanning(db.Model):
    __tablename__ = 'crop_planning'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    crop = db.Column(db.String(100))
    area = db.Column(db.Float)  # hectares
    planned_sowing = db.Column(db.Date)
    expected_harvest = db.Column(db.Date)
    status = db.Column(db.String(20), default='planned')  # planned, sown, growing, harvested
    
    user = db.relationship('User', backref=db.backref('crop_plans', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'crop': self.crop,
            'area': self.area,
            'planned_sowing': self.planned_sowing.isoformat() if self.planned_sowing else None,
            'expected_harvest': self.expected_harvest.isoformat() if self.expected_harvest else None,
            'status': self.status
        }


def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
