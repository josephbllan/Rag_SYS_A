"""
Database configuration and models for RAG system
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config.settings import DATABASE_URL

# Database setup 
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ShoeImage(Base):
    """Shoe image metadata table"""
    __tablename__ = "shoe_images"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), unique=True, index=True)
    original_path = Column(String(500))
    product_url = Column(String(1000))
    product_title = Column(Text)
    image_url = Column(String(1000))
    
    # Class characteristics
    pattern = Column(String(50), index=True)
    shape = Column(String(50), index=True)
    size = Column(String(50), index=True)
    brand = Column(String(50), index=True)
    
    # Additional metadata
    color = Column(String(50))
    style = Column(String(100))
    material = Column(String(100))
    price = Column(Float)
    
    # Technical metadata
    image_width = Column(Integer)
    image_height = Column(Integer)
    file_size = Column(Integer)
    format = Column(String(10))
    
    # Embeddings and vectors
    clip_embedding = Column(JSON)  # Store as JSON for flexibility
    resnet_features = Column(JSON)
    text_embedding = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    indexed_at = Column(DateTime)
    
    # Search metadata
    search_count = Column(Integer, default=0)
    last_searched = Column(DateTime)
    
    def __repr__(self):
        return f"<ShoeImage(id={self.id}, filename='{self.filename}', brand='{self.brand}')>"

class SearchQuery(Base):
    """Search query tracking table"""
    __tablename__ = "search_queries"
    
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text)
    query_type = Column(String(50))  # text, image, hybrid, metadata
    filters = Column(JSON)  # Applied filters
    results_count = Column(Integer)
    execution_time = Column(Float)  # in seconds
    
    # User/session info
    session_id = Column(String(100))
    user_id = Column(String(100))
    ip_address = Column(String(45))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SearchQuery(id={self.id}, query='{self.query_text[:50]}...', type='{self.query_type}')>"

class SearchResult(Base):
    """Search result tracking table"""
    __tablename__ = "search_results"
    
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, index=True)
    image_id = Column(Integer, index=True)
    similarity_score = Column(Float)
    rank = Column(Integer)
    
    # Result metadata
    clicked = Column(Integer, default=0)
    download_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SearchResult(id={self.id}, query_id={self.query_id}, image_id={self.image_id}, score={self.similarity_score})>"

class UserSession(Base):
    """User session tracking table"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True)
    user_id = Column(String(100))
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    # Session metadata
    total_searches = Column(Integer, default=0)
    total_clicks = Column(Integer, default=0)
    total_downloads = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, session_id='{self.session_id}', searches={self.total_searches})>"

class SystemMetrics(Base):
    """System performance metrics table"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), index=True)
    metric_value = Column(Float)
    metric_unit = Column(String(20))
    
    # Additional context
    context = Column(JSON)
    
    # Timestamps
    recorded_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SystemMetrics(id={self.id}, name='{self.metric_name}', value={self.metric_value})>"

# Database utility functions
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def drop_tables():
    """Drop all database tables"""
    Base.metadata.drop_all(bind=engine)

def reset_database():
    """Reset database by dropping and recreating tables"""
    drop_tables()
    create_tables()

# Database connection test
def test_connection():
    """Test database connection"""
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

if __name__ == "__main__":
    # Create tables when run directly
    create_tables()
    print("Database tables created successfully!")

