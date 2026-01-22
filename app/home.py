import os
import json
from flask import Flask, render_template, request, jsonify, Blueprint, send_from_directory
from app.poster_maker import generate_geo_data

home = Blueprint('home', __name__)

@home.route('/')
def hello_world():
    theme_options_path = os.path.join(os.path.dirname(__file__), 'static', 'themes')
    themes_list = []

    for theme_file in os.listdir(theme_options_path):
        if theme_file.endswith('.json'):
            theme_id = theme_file.replace('.json', '')
            percorso_completo = os.path.join(theme_options_path, theme_file)

            try:
                with open(percorso_completo, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    theme_name = data.get("name", theme_id)
                    themes_list.append({"id": theme_id, "name": theme_name})
            except Exception:
                continue

    return render_template('index.html', themes_list=themes_list)


@home.route("/generate", methods=["POST"])
def generate_map():
    data = request.json

    city = data.get("city")
    country = data.get("country")
    style = data.get("style")
    zoom_level = data.get("zoom", "city")

    # Map zoom levels to distances (in meters)
    zoom_map = {
        "village": 2000,
        "town": 5000,
        "city": 10000,
        "metropoli": 20000
    }
    dist = zoom_map.get(zoom_level, 10000)

    try:
        image_path = generate_geo_data(city, country, style, dist=dist)
        
        # Get filename only to serve via static route
        filename = os.path.basename(image_path)

        return jsonify({
            "status": "success",
            "image_url": f"/posters/{filename}"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@home.route("/posters/<path:filename>")
def serve_poster(filename):
    return send_from_directory("../posters", filename)




