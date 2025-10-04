"""
Web Interface for RAG System
Flask-based web application for interactive search
"""
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import os
import json
import logging
from pathlib import Path
import uuid
from datetime import datetime

from core.search_engine import create_search_engine
from core.query_processor import QueryProcessor
from config.settings import WEB_CONFIG, IMAGE_CONFIG, METADATA_CATEGORIES

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize search engine
search_engine = create_search_engine()
query_processor = QueryProcessor()

# Ensure upload directories exist
UPLOAD_FOLDER = Path("static/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# Allowed file extensions
ALLOWED_EXTENSIONS = set(IMAGE_CONFIG["supported_formats"])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in [ext[1:] for ext in ALLOWED_EXTENSIONS]

@app.route('/')
def index():
    """Main search page"""
    return render_template('index.html', categories=METADATA_CATEGORIES)

@app.route('/search')
def search():
    """Search page with results"""
    query = request.args.get('q', '')
    search_type = request.args.get('type', 'text')
    
    if not query:
        return redirect(url_for('index'))
    
    # Process search
    if search_type == 'text':
        results = search_engine.text_to_image_search(query, limit=20)
    elif search_type == 'image':
        results = search_engine.image_to_image_search(query, limit=20)
    else:
        results = search_engine.hybrid_search(query=query, limit=20)
    
    return render_template('search.html', 
                         query=query, 
                         search_type=search_type, 
                         results=results,
                         categories=METADATA_CATEGORIES)

@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for search"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        search_type = data.get('type', 'text')
        filters = data.get('filters', {})
        limit = data.get('limit', 20)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Process search based on type
        if search_type == 'text':
            results = search_engine.text_to_image_search(query, filters, limit)
        elif search_type == 'image':
            results = search_engine.image_to_image_search(query, filters, limit)
        elif search_type == 'hybrid':
            results = search_engine.hybrid_search(query=query, filters=filters, limit=limit)
        elif search_type == 'semantic':
            results = search_engine.semantic_search(query, filters, limit)
        else:
            return jsonify({'error': 'Invalid search type'}), 400
        
        return jsonify({
            'results': results,
            'total_count': len(results),
            'query': query,
            'type': search_type
        })
        
    except Exception as e:
        logger.error(f"API search failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """API endpoint for image upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            return jsonify({
                'filename': unique_filename,
                'path': filepath,
                'url': f'/static/uploads/{unique_filename}'
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/similar', methods=['POST'])
def api_similar():
    """API endpoint for finding similar images"""
    try:
        data = request.get_json()
        image_path = data.get('image_path', '')
        limit = data.get('limit', 10)
        
        if not image_path:
            return jsonify({'error': 'Image path is required'}), 400
        
        results = search_engine.image_to_image_search(image_path, limit=limit)
        
        return jsonify({
            'results': results,
            'total_count': len(results),
            'reference_image': image_path
        })
        
    except Exception as e:
        logger.error(f"Similar images search failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def api_recommendations():
    """API endpoint for recommendations"""
    try:
        data = request.get_json()
        image_path = data.get('image_path', '')
        limit = data.get('limit', 10)
        
        if not image_path:
            return jsonify({'error': 'Image path is required'}), 400
        
        results = search_engine.get_recommendations(image_path, limit=limit)
        
        return jsonify({
            'results': results,
            'total_count': len(results),
            'reference_image': image_path
        })
        
    except Exception as e:
        logger.error(f"Recommendations failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/natural', methods=['POST'])
def api_natural():
    """API endpoint for natural language queries"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Process natural language query
        intent = query_processor.process_query(query)
        
        # Execute search based on intent
        if intent.query_type == 'text':
            results = search_engine.text_to_image_search(
                query=" ".join(intent.search_terms),
                filters=intent.filters,
                limit=intent.limit
            )
        elif intent.query_type == 'image':
            if intent.image_path:
                results = search_engine.image_to_image_search(
                    image_path=intent.image_path,
                    filters=intent.filters,
                    limit=intent.limit
                )
            else:
                results = []
        elif intent.query_type == 'metadata':
            results = search_engine.metadata_search(
                filters=intent.filters,
                limit=intent.limit
            )
        else:  # hybrid
            results = search_engine.hybrid_search(
                query=" ".join(intent.search_terms),
                image_path=intent.image_path,
                filters=intent.filters,
                limit=intent.limit
            )
        
        return jsonify({
            'results': results,
            'total_count': len(results),
            'intent': {
                'type': intent.query_type,
                'search_terms': intent.search_terms,
                'filters': intent.filters,
                'image_path': intent.image_path
            },
            'original_query': query
        })
        
    except Exception as e:
        logger.error(f"Natural language search failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def api_stats():
    """API endpoint for search statistics"""
    try:
        stats = search_engine.get_search_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/categories')
def api_categories():
    """API endpoint for metadata categories"""
    return jsonify(METADATA_CATEGORIES)

@app.route('/browse')
def browse():
    """Browse all images"""
    try:
        # Get all images from database
        from config.database import get_db, ShoeImage
        db = next(get_db())
        images = db.query(ShoeImage).limit(100).all()
        
        return render_template('browse.html', images=images, categories=METADATA_CATEGORIES)
        
    except Exception as e:
        logger.error(f"Browse failed: {e}")
        return render_template('error.html', error=str(e))

@app.route('/image/<path:filename>')
def serve_image(filename):
    """Serve uploaded images"""
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    except Exception as e:
        logger.error(f"Image serve failed: {e}")
        return jsonify({'error': 'Image not found'}), 404

@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    try:
        stats = search_engine.get_search_stats()
        return render_template('analytics.html', stats=stats)
        
    except Exception as e:
        logger.error(f"Analytics failed: {e}")
        return render_template('error.html', error=str(e))

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error='Internal server error'), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Create static directory if it doesn't exist
    static_dir = Path('static')
    static_dir.mkdir(exist_ok=True)
    
    app.run(
        host=WEB_CONFIG['host'],
        port=WEB_CONFIG['port'],
        debug=WEB_CONFIG['debug']
    )

