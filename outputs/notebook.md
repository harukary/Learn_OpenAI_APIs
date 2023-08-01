```python
# !pip install -U googlemaps
# !pip install folium
# !pip install python-dotenv
```

## Preparing


```python
from dotenv import find_dotenv,load_dotenv
load_dotenv(find_dotenv())
```




    True




```python
import os,json
from datetime import datetime
import numpy as np
import folium
from branca.element import Figure
import googlemaps
gmaps = googlemaps.Client(key=os.environ['GOOGLE_MAPS_API_KEY'])
```

## Get data from address


```python
# Geocoding an address
address = '1600 Amphitheatre Parkway, Mountain View, CA'
geocode_result = gmaps.geocode(address)
print(json.dumps(geocode_result,indent=2))
```

    [
      {
        "address_components": [
          {
            "long_name": "Google Building 40",
            "short_name": "Google Building 40",
            "types": [
              "premise"
            ]
          },
          {
            "long_name": "1600",
            "short_name": "1600",
            "types": [
              "street_number"
            ]
          },
          {
            "long_name": "Amphitheatre Parkway",
            "short_name": "Amphitheatre Pkwy",
            "types": [
              "route"
            ]
          },
          {
            "long_name": "Mountain View",
            "short_name": "Mountain View",
            "types": [
              "locality",
              "political"
            ]
          },
          {
            "long_name": "Santa Clara County",
            "short_name": "Santa Clara County",
            "types": [
              "administrative_area_level_2",
              "political"
            ]
          },
          {
            "long_name": "California",
            "short_name": "CA",
            "types": [
              "administrative_area_level_1",
              "political"
            ]
          },
          {
            "long_name": "United States",
            "short_name": "US",
            "types": [
              "country",
              "political"
            ]
          },
          {
            "long_name": "94043",
            "short_name": "94043",
            "types": [
              "postal_code"
            ]
          }
        ],
        "formatted_address": "Google Building 40, 1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA",
        "geometry": {
          "bounds": {
            "northeast": {
              "lat": 37.4226618,
              "lng": -122.0829302
            },
            "southwest": {
              "lat": 37.4220699,
              "lng": -122.084958
            }
          },
          "location": {
            "lat": 37.4223878,
            "lng": -122.0841877
          },
          "location_type": "ROOFTOP",
          "viewport": {
            "northeast": {
              "lat": 37.42372298029149,
              "lng": -122.0825951197085
            },
            "southwest": {
              "lat": 37.4210250197085,
              "lng": -122.0852930802915
            }
          }
        },
        "place_id": "ChIJj38IfwK6j4ARNcyPDnEGa9g",
        "types": [
          "premise"
        ]
      }
    ]



```python
loc = geocode_result[0]['geometry']['location']
map = folium.Map(location=[loc['lat'],loc['lng']], zoom_start=14)
folium.Marker(location=[loc['lat'],loc['lng']], popup=address).add_to(map)
fig = Figure(width=800, height=600)
fig.add_child(map)
```




<iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;

        &lt;script&gt;
            L_NO_TOUCH = false;
            L_DISABLE_3D = false;
        &lt;/script&gt;

    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-1.12.4.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_41a8b1c64ee2855d66a961b1552ac2dc {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_41a8b1c64ee2855d66a961b1552ac2dc&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_41a8b1c64ee2855d66a961b1552ac2dc = L.map(
                &quot;map_41a8b1c64ee2855d66a961b1552ac2dc&quot;,
                {
                    center: [37.4223878, -122.0841877],
                    crs: L.CRS.EPSG3857,
                    zoom: 14,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );





            var tile_layer_cecb422b03d65e377d2cf00c30a1aff2 = L.tileLayer(
                &quot;https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;Data by \u0026copy; \u003ca target=\&quot;_blank\&quot; href=\&quot;http://openstreetmap.org\&quot;\u003eOpenStreetMap\u003c/a\u003e, under \u003ca target=\&quot;_blank\&quot; href=\&quot;http://www.openstreetmap.org/copyright\&quot;\u003eODbL\u003c/a\u003e.&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 18, &quot;maxZoom&quot;: 18, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            ).addTo(map_41a8b1c64ee2855d66a961b1552ac2dc);


            var marker_65ae6789400442c0ead45addb71f74e6 = L.marker(
                [37.4223878, -122.0841877],
                {}
            ).addTo(map_41a8b1c64ee2855d66a961b1552ac2dc);


        var popup_5bf25214407fa537994ede245c135223 = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_412105e7786d77c8ce787c7cd613f2c6 = $(`&lt;div id=&quot;html_412105e7786d77c8ce787c7cd613f2c6&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;1600 Amphitheatre Parkway, Mountain View, CA&lt;/div&gt;`)[0];
                popup_5bf25214407fa537994ede245c135223.setContent(html_412105e7786d77c8ce787c7cd613f2c6);



        marker_65ae6789400442c0ead45addb71f74e6.bindPopup(popup_5bf25214407fa537994ede245c135223)
        ;



&lt;/script&gt;
&lt;/html&gt;" width="800" height="600"style="border:none !important;" "allowfullscreen" "webkitallowfullscreen" "mozallowfullscreen"></iframe>


