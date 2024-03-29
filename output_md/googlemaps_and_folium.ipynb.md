﻿```python
# !pip install -U googlemaps
# !pip install folium
# !pip install python-dotenv
```


```python
from dotenv import find_dotenv,load_dotenv
load_dotenv(find_dotenv())
```




    True



## Preparing


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
print(geocode_result[0]['geometry']['location'])
```

    {'lat': 37.4223878, 'lng': -122.0841877}



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
                #map_ce9189c0c3ef45fee554ac469b9eea93 {
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


            &lt;div class=&quot;folium-map&quot; id=&quot;map_ce9189c0c3ef45fee554ac469b9eea93&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_ce9189c0c3ef45fee554ac469b9eea93 = L.map(
                &quot;map_ce9189c0c3ef45fee554ac469b9eea93&quot;,
                {
                    center: [37.4223878, -122.0841877],
                    crs: L.CRS.EPSG3857,
                    zoom: 14,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );





            var tile_layer_c8ae54063cd340123a2dbeb9e3e6976f = L.tileLayer(
                &quot;https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;Data by \u0026copy; \u003ca target=\&quot;_blank\&quot; href=\&quot;http://openstreetmap.org\&quot;\u003eOpenStreetMap\u003c/a\u003e, under \u003ca target=\&quot;_blank\&quot; href=\&quot;http://www.openstreetmap.org/copyright\&quot;\u003eODbL\u003c/a\u003e.&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 18, &quot;maxZoom&quot;: 18, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            ).addTo(map_ce9189c0c3ef45fee554ac469b9eea93);


            var marker_e65f2e409b8864e271bd34690a0f196a = L.marker(
                [37.4223878, -122.0841877],
                {}
            ).addTo(map_ce9189c0c3ef45fee554ac469b9eea93);


        var popup_60a2d8ae270a7412c9a50844b292d65a = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_d9966c52becabc3f62a005ea64b84042 = $(`&lt;div id=&quot;html_d9966c52becabc3f62a005ea64b84042&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;1600 Amphitheatre Parkway, Mountain View, CA&lt;/div&gt;`)[0];
                popup_60a2d8ae270a7412c9a50844b292d65a.setContent(html_d9966c52becabc3f62a005ea64b84042);



        marker_e65f2e409b8864e271bd34690a0f196a.bindPopup(popup_60a2d8ae270a7412c9a50844b292d65a)
        ;



&lt;/script&gt;
&lt;/html&gt;" width="800" height="600"style="border:none !important;" "allowfullscreen" "webkitallowfullscreen" "mozallowfullscreen"></iframe>



## Get Address from lat and lng


```python
# Look up an address with reverse geocoding
reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))
print(json.dumps(reverse_geocode_result,indent=2))
```

    [
      {
        "address_components": [
          {
            "long_name": "277",
            "short_name": "277",
            "types": [
              "street_number"
            ]
          },
          {
            "long_name": "Bedford Avenue",
            "short_name": "Bedford Ave",
            "types": [
              "route"
            ]
          },
          {
            "long_name": "Williamsburg",
            "short_name": "Williamsburg",
            "types": [
              "neighborhood",
              "political"
            ]
          },
          {
            "long_name": "Brooklyn",
            "short_name": "Brooklyn",
            "types": [
              "political",
              "sublocality",
              "sublocality_level_1"
            ]
          },
          {
            "long_name": "Kings County",
            "short_name": "Kings County",
            "types": [
              "administrative_area_level_2",
              "political"
            ]
          },
          {
            "long_name": "New York",
            "short_name": "NY",
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
            "long_name": "11211",
            "short_name": "11211",
            "types": [
              "postal_code"
            ]
          }
        ],
        "formatted_address": "277 Bedford Ave, Brooklyn, NY 11211, USA",
        "geometry": {
          "location": {
            "lat": 40.7142205,
            "lng": -73.9612903
          },
          "location_type": "ROOFTOP",
          "viewport": {
            "northeast": {
              "lat": 40.71556948029149,
              "lng": -73.95994131970849
            },
            "southwest": {
              "lat": 40.7128715197085,
              "lng": -73.9626392802915
            }
          }
        },
        "place_id": "ChIJd8BlQ2BZwokRAFUEcm_qrcA",
        "plus_code": {
          "compound_code": "P27Q+MF New York, NY, USA",
          "global_code": "87G8P27Q+MF"
        },
        "types": [
          "street_address"
        ]
      },
      {
        "address_components": [
          {
            "long_name": "279",
            "short_name": "279",
            "types": [
              "street_number"
            ]
          },
          {
            "long_name": "Bedford Avenue",
            "short_name": "Bedford Ave",
            "types": [
              "route"
            ]
          },
          {
            "long_name": "Williamsburg",
            "short_name": "Williamsburg",
            "types": [
              "neighborhood",
              "political"
            ]
          },
          {
            "long_name": "Brooklyn",
            "short_name": "Brooklyn",
            "types": [
              "political",
              "sublocality",
              "sublocality_level_1"
            ]
          },
          {
            "long_name": "Kings County",
            "short_name": "Kings County",
            "types": [
              "administrative_area_level_2",
              "political"
            ]
          },
          {
            "long_name": "New York",
            "short_name": "NY",
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
            "long_name": "11211",
            "short_name": "11211",
            "types": [
              "postal_code"
            ]
          },
          {
            "long_name": "4203",
            "short_name": "4203",
            "types": [
              "postal_code_suffix"
            ]
          }
        ],
        "formatted_address": "279 Bedford Ave, Brooklyn, NY 11211, USA",
        "geometry": {
          "bounds": {
            "northeast": {
              "lat": 40.7142628,
              "lng": -73.9612131
            },
            "southwest": {
              "lat": 40.7141534,
              "lng": -73.9613792
            }
          },
          "location": {
            "lat": 40.7142015,
            "lng": -73.96130769999999
          },
          "location_type": "ROOFTOP",
          "viewport": {
            "northeast": {
              "lat": 40.7155570802915,
              "lng": -73.95994716970849
            },
            "southwest": {
              "lat": 40.7128591197085,
              "lng": -73.96264513029149
            }
          }
        },
        "place_id": "ChIJRYYERGBZwokRAM4n1GlcYX4",
        "types": [
          "premise"
        ]
      },
      {
        "address_components": [
          {
            "long_name": "277",
            "short_name": "277",
            "types": [
              "street_number"
            ]
          },
          {
            "long_name": "Bedford Avenue",
            "short_name": "Bedford Ave",
            "types": [
              "route"
            ]
          },
          {
            "long_name": "Williamsburg",
            "short_name": "Williamsburg",
            "types": [
              "neighborhood",
              "political"
            ]
          },
          {
            "long_name": "Brooklyn",
            "short_name": "Brooklyn",
            "types": [
              "political",
              "sublocality",
              "sublocality_level_1"
            ]
          },
          {
            "long_name": "Kings County",
            "short_name": "Kings County",
            "types": [
              "administrative_area_level_2",
              "political"
            ]
          },
          {
            "long_name": "New York",
            "short_name": "NY",
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
            "long_name": "11211",
            "short_name": "11211",
            "types": [
              "postal_code"
            ]
          }
        ],
        "formatted_address": "277 Bedford Ave, Brooklyn, NY 11211, USA",
        "geometry": {
          "location": {
            "lat": 40.7142205,
            "lng": -73.9612903
          },
          "location_type": "ROOFTOP",
          "viewport": {
            "northeast": {
              "lat": 40.71556948029149,
              "lng": -73.95994131970849
            },
            "southwest": {
              "lat": 40.7128715197085,
              "lng": -73.9626392802915
            }
          }
        },
        "place_id": "ChIJF0hlQ2BZwokRsrY2RAlFbAE",
        "plus_code": {
          "compound_code": "P27Q+MF Brooklyn, NY, USA",
          "global_code": "87G8P27Q+MF"
        },
        "types": [
          "establishment",
          "point_of_interest"
        ]
      },
      {
        "address_components": [
          {
            "long_name": "291-275",
            "short_name": "291-275",
            "types": [
              "street_number"
            ]
          },
          {
            "long_name": "Bedford Avenue",
            "short_name": "Bedford Ave",
            "types": [
              "route"
            ]
          },
          {
            "long_name": "Williamsburg",
            "short_name": "Williamsburg",
            "types": [
              "neighborhood",
              "political"
            ]
          },
          {
            "long_name": "Brooklyn",
            "short_name": "Brooklyn",
            "types": [
              "political",
              "sublocality",
              "sublocality_level_1"
            ]
          },
          {
            "long_name": "Kings County",
            "short_name": "Kings County",
            "types": [
              "administrative_area_level_2",
              "political"
            ]
          },
          {
            "long_name": "New York",
            "short_name": "NY",
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
            "long_name": "11211",
            "short_name": "11211",
            "types": [
              "postal_code"
            ]
          }
        ],
        "formatted_address": "291-275 Bedford Ave, Brooklyn, NY 11211, USA",
        "geometry": {
          "bounds": {
            "northeast": {
              "lat": 40.7145065,
              "lng": -73.9612923
            },
            "southwest": {
              "lat": 40.7139055,
              "lng": -73.96168349999999
            }
          },
          "location": {
            "lat": 40.7142045,
            "lng": -73.9614845
          },
          "location_type": "GEOMETRIC_CENTER",
          "viewport": {
            "northeast": {
              "lat": 40.7155549802915,
              "lng": -73.96013891970848
            },
            "southwest": {
              "lat": 40.7128570197085,
              "lng": -73.96283688029149
            }
          }
        },
        "place_id": "ChIJ8ThWRGBZwokR3E1zUisk3LU",
        "types": [
          "route"
        ]
      },
      {
        "address_components": [
          {
            "long_name": "P27Q+MC",
            "short_name": "P27Q+MC",
            "types": [
              "plus_code"
            ]
          },
          {
            "long_name": "New York",
            "short_name": "New York",
            "types": [
              "locality",
              "political"
            ]
          },
          {
            "long_name": "New York",
            "short_name": "NY",
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
          }
        ],
        "formatted_address": "P27Q+MC New York, NY, USA",
        "geometry": {
          "bounds": {
            "northeast": {
              "lat": 40.71425,
              "lng": -73.96137499999999
            },
            "southwest": {
              "lat": 40.714125,
              "lng": -73.9615
            }
          },
          "location": {
            "lat": 40.714224,
            "lng": -73.961452
          },
          "location_type": "GEOMETRIC_CENTER",
          "viewport": {
            "northeast": {
              "lat": 40.71553648029149,
              "lng": -73.96008851970849
            },
            "southwest": {
              "lat": 40.71283851970849,
              "lng": -73.96278648029151
            }
          }
        },
        "place_id": "GhIJWAIpsWtbREARHyv4bYh9UsA",
        "plus_code": {
          "compound_code": "P27Q+MC New York, NY, USA",
          "global_code": "87G8P27Q+MC"
        },
        "types": [
          "plus_code"
        ]
      },
      {
        "address_components": [
          {
            "long_name": "South Williamsburg",
            "short_name": "South Williamsburg",
            "types": [
              "neighborhood",
              "political"
            ]
          },
          {
            "long_name": "Brooklyn",
            "short_name": "Brooklyn",
            "types": [
              "political",
              "sublocality",
              "sublocality_level_1"
            ]
          },
          {
            "long_name": "Kings County",
            "short_name": "Kings County",
            "types": [
              "administrative_area_level_2",
              "political"
            ]
          },
          {
            "long_name": "New York",
            "short_name": "NY",
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
          }
        ],
        "formatted_address": "South Williamsburg, Brooklyn, NY, USA",
        "geometry": {
          "bounds": {
            "northeast": {
              "lat": 40.7167119,
              "lng": -73.9420904
            },
            "southwest": {
              "lat": 40.6984866,
              "lng": -73.9699432
            }
          },
          "location": {
            "lat": 40.7043921,
            "lng": -73.9565551
          },
          "location_type": "APPROXIMATE",
          "viewport": {
            "northeast": {
              "lat": 40.7167119,
              "lng": -73.9420904
            },
            "southwest": {
              "lat": 40.6984866,
              "lng": -73.9699432
            }
          }
        },
        "place_id": "ChIJR3_ODdlbwokRYtN19kNtcuk",
        "types": [
          "neighborhood",
          "political"
        ]
      },
      {
        "address_components": [
          {
            "long_name": "11211",
            "short_name": "11211",
            "types": [
              "postal_code"
            ]
          },
          {
            "long_name": "Brooklyn",
            "short_name": "Brooklyn",
            "types": [
              "political",
              "sublocality",
              "sublocality_level_1"
            ]
          },
          {
            "long_name": "New York",
            "short_name": "New York",
            "types": [
              "locality",
              "political"
            ]
          },
          {
            "long_name": "New York",
            "short_name": "NY",
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
          }
        ],
        "formatted_address": "Brooklyn, NY 11211, USA",
        "geometry": {
          "bounds": {
            "northeast": {
              "lat": 40.7280089,
              "lng": -73.9207299
            },
            "southwest": {
              "lat": 40.7008331,
              "lng": -73.9644697
            }
          },
          "location": {
            "lat": 40.7093358,
            "lng": -73.9565551
          },
          "location_type": "APPROXIMATE",
          "viewport": {
            "northeast": {
              "lat": 40.7280089,
              "lng": -73.9207299
            },
            "southwest": {
              "lat": 40.7008331,
              "lng": -73.9644697
            }
          }
        },
        "place_id": "ChIJvbEjlVdZwokR4KapM3WCFRw",
        "types": [
          "postal_code"
        ]
      },
      {
        "address_components": [
          {
            "long_name": "Williamsburg",
            "short_name": "Williamsburg",
            "types": [
              "neighborhood",
              "political"
            ]
          },
          {
            "long_name": "Brooklyn",
            "short_name": "Brooklyn",
            "types": [
              "political",
              "sublocality",
              "sublocality_level_1"
            ]
          },
          {
            "long_name": "Kings County",
            "short_name": "Kings County",
            "types": [
              "administrative_area_level_2",
              "political"
            ]
          },
          {
            "long_name": "New York",
            "short_name": "NY",
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
          }
        ],
        "formatted_address": "Williamsburg, Brooklyn, NY, USA",
        "geometry": {
          "bounds": {
            "northeast": {
              "lat": 40.7251773,
              "lng": -73.936498
            },
            "southwest": {
              "lat": 40.6979329,
              "lng": -73.96984499999999
            }
          },
          "location": {
            "lat": 40.7081156,
            "lng": -73.9570696
          },
          "location_type": "APPROXIMATE",
          "viewport": {
            "northeast": {
              "lat": 40.7251773,
              "lng": -73.936498
            },
            "southwest": {
              "lat": 40.6979329,
              "lng": -73.96984499999999
            }
          }
        },
        "place_id": "ChIJQSrBBv1bwokRbNfFHCnyeYI",
        "types": [
          "neighborhood",
          "political"
        ]
      },
      {
        "address_components": [
          {
            "long_name": "Kings County",
            "short_name": "Kings County",
            "types": [
              "administrative_area_level_2",
              "political"
            ]
          },
          {
            "long_name": "Brooklyn",
            "short_name": "Brooklyn",
            "types": [
              "political",
              "sublocality",
              "sublocality_level_1"
            ]
          },
          {
            "long_name": "New York",
            "short_name": "NY",
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
          }
        ],
        "formatted_address": "Kings County, Brooklyn, NY, USA",
        "geometry": {
          "bounds": {
            "northeast": {
              "lat": 40.7394209,
              "lng": -73.8330411
            },
            "southwest": {
              "lat": 40.551042,
              "lng": -74.05663
            }
          },
          "location": {
            "lat": 40.6528762,
            "lng": -73.95949399999999
          },
          "location_type": "APPROXIMATE",
          "viewport": {
            "northeast": {
              "lat": 40.7394209,
              "lng": -73.8330411
            },
            "southwest": {
              "lat": 40.551042,
              "lng": -74.05663
            }
          }
        },
        "place_id": "ChIJOwE7_GTtwokRs75rhW4_I6M",
        "types": [
          "administrative_area_level_2",
          "political"
        ]
      },
      {
        "address_components": [
          {
            "long_name": "Brooklyn",
            "short_name": "Brooklyn",
            "types": [
              "political",
              "sublocality",
              "sublocality_level_1"
            ]
          },
          {
            "long_name": "Kings County",
            "short_name": "Kings County",
            "types": [
              "administrative_area_level_2",
              "political"
            ]
          },
          {
            "long_name": "New York",
            "short_name": "NY",
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
          }
        ],
        "formatted_address": "Brooklyn, NY, USA",
        "geometry": {
          "bounds": {
            "northeast": {
              "lat": 40.739446,
              "lng": -73.8333651
            },
            "southwest": {
              "lat": 40.551042,
              "lng": -74.05663
            }
          },
          "location": {
            "lat": 40.6781784,
            "lng": -73.9441579
          },
          "location_type": "APPROXIMATE",
          "viewport": {
            "northeast": {
              "lat": 40.739446,
              "lng": -73.8333651
            },
            "southwest": {
              "lat": 40.551042,
              "lng": -74.05663
            }
          }
        },
        "place_id": "ChIJCSF8lBZEwokRhngABHRcdoI",
        "types": [
          "political",
          "sublocality",
          "sublocality_level_1"
        ]
      },
      {
        "address_components": [
          {
            "long_name": "New York",
            "short_name": "New York",
            "types": [
              "locality",
              "political"
            ]
          },
          {
            "long_name": "New York",
            "short_name": "NY",
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
          }
        ],
        "formatted_address": "New York, NY, USA",
        "geometry": {
          "bounds": {
            "northeast": {
              "lat": 40.9175771,
              "lng": -73.70027209999999
            },
            "southwest": {
              "lat": 40.4773991,
              "lng": -74.25908989999999
            }
          },
          "location": {
            "lat": 40.7127753,
            "lng": -74.0059728
          },
          "location_type": "APPROXIMATE",
          "viewport": {
            "northeast": {
              "lat": 40.9175771,
              "lng": -73.70027209999999
            },
            "southwest": {
              "lat": 40.4773991,
              "lng": -74.25908989999999
            }
          }
        },
        "place_id": "ChIJOwg_06VPwokRYv534QaPC8g",
        "types": [
          "locality",
          "political"
        ]
      },
      {
        "address_components": [
          {
            "long_name": "New York",
            "short_name": "NY",
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
          }
        ],
        "formatted_address": "New York, USA",
        "geometry": {
          "bounds": {
            "northeast": {
              "lat": 45.015861,
              "lng": -71.777491
            },
            "southwest": {
              "lat": 40.476578,
              "lng": -79.7625901
            }
          },
          "location": {
            "lat": 43.2994285,
            "lng": -74.21793260000001
          },
          "location_type": "APPROXIMATE",
          "viewport": {
            "northeast": {
              "lat": 45.015861,
              "lng": -71.777491
            },
            "southwest": {
              "lat": 40.476578,
              "lng": -79.7625901
            }
          }
        },
        "place_id": "ChIJqaUj8fBLzEwRZ5UY3sHGz90",
        "types": [
          "administrative_area_level_1",
          "political"
        ]
      },
      {
        "address_components": [
          {
            "long_name": "United States",
            "short_name": "US",
            "types": [
              "country",
              "political"
            ]
          }
        ],
        "formatted_address": "United States",
        "geometry": {
          "bounds": {
            "northeast": {
              "lat": 74.071038,
              "lng": -66.885417
            },
            "southwest": {
              "lat": 18.7763,
              "lng": 166.9999999
            }
          },
          "location": {
            "lat": 37.09024,
            "lng": -95.712891
          },
          "location_type": "APPROXIMATE",
          "viewport": {
            "northeast": {
              "lat": 74.071038,
              "lng": -66.885417
            },
            "southwest": {
              "lat": 18.7763,
              "lng": 166.9999999
            }
          }
        },
        "place_id": "ChIJCzYy5IS16lQRQrfeQ5K5Oxw",
        "types": [
          "country",
          "political"
        ]
      }
    ]



```python
len(reverse_geocode_result)
locs = [
    [res['geometry']['location']['lat'],res['geometry']['location']['lng']]
    for res in reverse_geocode_result
]
map = folium.Map(location=list(np.mean(locs,axis=0)), zoom_start=5)
for loc,res in zip(locs,reverse_geocode_result):
    # print(loc)
    folium.Marker(location=loc, popup=res['formatted_address']).add_to(map)
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
                #map_8ef04dd3cfd9d6a70e87532c3fa837aa {
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


            &lt;div class=&quot;folium-map&quot; id=&quot;map_8ef04dd3cfd9d6a70e87532c3fa837aa&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_8ef04dd3cfd9d6a70e87532c3fa837aa = L.map(
                &quot;map_8ef04dd3cfd9d6a70e87532c3fa837aa&quot;,
                {
                    center: [40.625108684615384, -75.6551886846154],
                    crs: L.CRS.EPSG3857,
                    zoom: 5,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );





            var tile_layer_15133272be99bcb1d697476f679c51eb = L.tileLayer(
                &quot;https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;Data by \u0026copy; \u003ca target=\&quot;_blank\&quot; href=\&quot;http://openstreetmap.org\&quot;\u003eOpenStreetMap\u003c/a\u003e, under \u003ca target=\&quot;_blank\&quot; href=\&quot;http://www.openstreetmap.org/copyright\&quot;\u003eODbL\u003c/a\u003e.&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 18, &quot;maxZoom&quot;: 18, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            ).addTo(map_8ef04dd3cfd9d6a70e87532c3fa837aa);


            var marker_434bf07d9294255b8eaec8787fab92f9 = L.marker(
                [40.7142205, -73.9612903],
                {}
            ).addTo(map_8ef04dd3cfd9d6a70e87532c3fa837aa);


        var popup_6fa8cfc12a45f06fc9612a684d48c44f = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_b7f0465a1c97dfe38ae263e58ed49f20 = $(`&lt;div id=&quot;html_b7f0465a1c97dfe38ae263e58ed49f20&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;277 Bedford Ave, Brooklyn, NY 11211, USA&lt;/div&gt;`)[0];
                popup_6fa8cfc12a45f06fc9612a684d48c44f.setContent(html_b7f0465a1c97dfe38ae263e58ed49f20);



        marker_434bf07d9294255b8eaec8787fab92f9.bindPopup(popup_6fa8cfc12a45f06fc9612a684d48c44f)
        ;




            var marker_74e7eca32d54b18e34cd7ee63c2e4955 = L.marker(
                [40.7142015, -73.96130769999999],
                {}
            ).addTo(map_8ef04dd3cfd9d6a70e87532c3fa837aa);


        var popup_393e9384381c9199a530422cbb7431a0 = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_9ef5b9bd3acd7da6c2ca8c71aaf48dfa = $(`&lt;div id=&quot;html_9ef5b9bd3acd7da6c2ca8c71aaf48dfa&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;279 Bedford Ave, Brooklyn, NY 11211, USA&lt;/div&gt;`)[0];
                popup_393e9384381c9199a530422cbb7431a0.setContent(html_9ef5b9bd3acd7da6c2ca8c71aaf48dfa);



        marker_74e7eca32d54b18e34cd7ee63c2e4955.bindPopup(popup_393e9384381c9199a530422cbb7431a0)
        ;




            var marker_075126a52ea2763f4fdcbe18b853fef8 = L.marker(
                [40.7142205, -73.9612903],
                {}
            ).addTo(map_8ef04dd3cfd9d6a70e87532c3fa837aa);


        var popup_1d387be3007b7992489892e2e5e34d37 = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_5e7658c88670001c80ddd4bfde70c2a0 = $(`&lt;div id=&quot;html_5e7658c88670001c80ddd4bfde70c2a0&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;277 Bedford Ave, Brooklyn, NY 11211, USA&lt;/div&gt;`)[0];
                popup_1d387be3007b7992489892e2e5e34d37.setContent(html_5e7658c88670001c80ddd4bfde70c2a0);



        marker_075126a52ea2763f4fdcbe18b853fef8.bindPopup(popup_1d387be3007b7992489892e2e5e34d37)
        ;




            var marker_92228917be88fa7dd1519e43cdb8e9ac = L.marker(
                [40.7142045, -73.9614845],
                {}
            ).addTo(map_8ef04dd3cfd9d6a70e87532c3fa837aa);


        var popup_da5814fecd3e3e60b070757cac649ab9 = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_9c745dedb485fe93dcf24c25379af23e = $(`&lt;div id=&quot;html_9c745dedb485fe93dcf24c25379af23e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;291-275 Bedford Ave, Brooklyn, NY 11211, USA&lt;/div&gt;`)[0];
                popup_da5814fecd3e3e60b070757cac649ab9.setContent(html_9c745dedb485fe93dcf24c25379af23e);



        marker_92228917be88fa7dd1519e43cdb8e9ac.bindPopup(popup_da5814fecd3e3e60b070757cac649ab9)
        ;




            var marker_ad2f096530a7d9c9521695928314ae5a = L.marker(
                [40.714224, -73.961452],
                {}
            ).addTo(map_8ef04dd3cfd9d6a70e87532c3fa837aa);


        var popup_fa20c8b3911067d0c4da6caff777e0f8 = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_588426180b0126e52a4fa6522d62fa4a = $(`&lt;div id=&quot;html_588426180b0126e52a4fa6522d62fa4a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;P27Q+MC New York, NY, USA&lt;/div&gt;`)[0];
                popup_fa20c8b3911067d0c4da6caff777e0f8.setContent(html_588426180b0126e52a4fa6522d62fa4a);



        marker_ad2f096530a7d9c9521695928314ae5a.bindPopup(popup_fa20c8b3911067d0c4da6caff777e0f8)
        ;




            var marker_362f6d374f4371d7d16015389156cdd0 = L.marker(
                [40.7043921, -73.9565551],
                {}
            ).addTo(map_8ef04dd3cfd9d6a70e87532c3fa837aa);


        var popup_042a49c94541ac9f3674fe2e48544710 = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_694f8da0f85ce73c789e22b6217b0709 = $(`&lt;div id=&quot;html_694f8da0f85ce73c789e22b6217b0709&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;South Williamsburg, Brooklyn, NY, USA&lt;/div&gt;`)[0];
                popup_042a49c94541ac9f3674fe2e48544710.setContent(html_694f8da0f85ce73c789e22b6217b0709);



        marker_362f6d374f4371d7d16015389156cdd0.bindPopup(popup_042a49c94541ac9f3674fe2e48544710)
        ;




            var marker_48f330087f0a463043ddea74c07f4237 = L.marker(
                [40.7093358, -73.9565551],
                {}
            ).addTo(map_8ef04dd3cfd9d6a70e87532c3fa837aa);


        var popup_c232b6234c20ed9a065468845b680346 = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_f415722986e65ba0e6b54e58abb0ef01 = $(`&lt;div id=&quot;html_f415722986e65ba0e6b54e58abb0ef01&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;Brooklyn, NY 11211, USA&lt;/div&gt;`)[0];
                popup_c232b6234c20ed9a065468845b680346.setContent(html_f415722986e65ba0e6b54e58abb0ef01);



        marker_48f330087f0a463043ddea74c07f4237.bindPopup(popup_c232b6234c20ed9a065468845b680346)
        ;




            var marker_e9d03d79cfc0df7622fc9212df4bb74a = L.marker(
                [40.7081156, -73.9570696],
                {}
            ).addTo(map_8ef04dd3cfd9d6a70e87532c3fa837aa);


        var popup_787e0c7698b58a6bc4d2cc27488168ae = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_f0d03cd2c1ce2be3ce57102538fe2d91 = $(`&lt;div id=&quot;html_f0d03cd2c1ce2be3ce57102538fe2d91&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;Williamsburg, Brooklyn, NY, USA&lt;/div&gt;`)[0];
                popup_787e0c7698b58a6bc4d2cc27488168ae.setContent(html_f0d03cd2c1ce2be3ce57102538fe2d91);



        marker_e9d03d79cfc0df7622fc9212df4bb74a.bindPopup(popup_787e0c7698b58a6bc4d2cc27488168ae)
        ;




            var marker_f76ac3882b98626b5ac3fe7cb46c13a4 = L.marker(
                [40.6528762, -73.95949399999999],
                {}
            ).addTo(map_8ef04dd3cfd9d6a70e87532c3fa837aa);


        var popup_e338adb6e0377c6dfb644ba8a4884e1d = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_bbad91e5b78d1fdd90201f5070e38017 = $(`&lt;div id=&quot;html_bbad91e5b78d1fdd90201f5070e38017&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;Kings County, Brooklyn, NY, USA&lt;/div&gt;`)[0];
                popup_e338adb6e0377c6dfb644ba8a4884e1d.setContent(html_bbad91e5b78d1fdd90201f5070e38017);



        marker_f76ac3882b98626b5ac3fe7cb46c13a4.bindPopup(popup_e338adb6e0377c6dfb644ba8a4884e1d)
        ;




            var marker_333c2aa9e585a5a2a1428d03376dc9dc = L.marker(
                [40.6781784, -73.9441579],
                {}
            ).addTo(map_8ef04dd3cfd9d6a70e87532c3fa837aa);


        var popup_abd62adf5bc28e2a2b918ef4016352a1 = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_285f4de2b3fe888ab8a2c27b14e8e4ca = $(`&lt;div id=&quot;html_285f4de2b3fe888ab8a2c27b14e8e4ca&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;Brooklyn, NY, USA&lt;/div&gt;`)[0];
                popup_abd62adf5bc28e2a2b918ef4016352a1.setContent(html_285f4de2b3fe888ab8a2c27b14e8e4ca);



        marker_333c2aa9e585a5a2a1428d03376dc9dc.bindPopup(popup_abd62adf5bc28e2a2b918ef4016352a1)
        ;




            var marker_ac9951b0c6874ee6560517691b641942 = L.marker(
                [40.7127753, -74.0059728],
                {}
            ).addTo(map_8ef04dd3cfd9d6a70e87532c3fa837aa);


        var popup_dbd147bac59dd1c23aa817b6f1d115fd = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_1707f5516410019196ff5a579db190bf = $(`&lt;div id=&quot;html_1707f5516410019196ff5a579db190bf&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;New York, NY, USA&lt;/div&gt;`)[0];
                popup_dbd147bac59dd1c23aa817b6f1d115fd.setContent(html_1707f5516410019196ff5a579db190bf);



        marker_ac9951b0c6874ee6560517691b641942.bindPopup(popup_dbd147bac59dd1c23aa817b6f1d115fd)
        ;




            var marker_35e8b7308741b5c241a3b5dc26c01944 = L.marker(
                [43.2994285, -74.21793260000001],
                {}
            ).addTo(map_8ef04dd3cfd9d6a70e87532c3fa837aa);


        var popup_092935a6cb8f1c74169de2fcb005f3bf = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_6270bc89d140bd2cd14772cf4145bbbe = $(`&lt;div id=&quot;html_6270bc89d140bd2cd14772cf4145bbbe&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;New York, USA&lt;/div&gt;`)[0];
                popup_092935a6cb8f1c74169de2fcb005f3bf.setContent(html_6270bc89d140bd2cd14772cf4145bbbe);



        marker_35e8b7308741b5c241a3b5dc26c01944.bindPopup(popup_092935a6cb8f1c74169de2fcb005f3bf)
        ;




            var marker_02c9d7b33d17b80148408e03333f6284 = L.marker(
                [37.09024, -95.712891],
                {}
            ).addTo(map_8ef04dd3cfd9d6a70e87532c3fa837aa);


        var popup_c1c117e3bfa0e8e6be824892041ea8da = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_d6263a4e4589e05c5e698b44f7887bbd = $(`&lt;div id=&quot;html_d6263a4e4589e05c5e698b44f7887bbd&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;United States&lt;/div&gt;`)[0];
                popup_c1c117e3bfa0e8e6be824892041ea8da.setContent(html_d6263a4e4589e05c5e698b44f7887bbd);



        marker_02c9d7b33d17b80148408e03333f6284.bindPopup(popup_c1c117e3bfa0e8e6be824892041ea8da)
        ;



&lt;/script&gt;
&lt;/html&gt;" width="800" height="600"style="border:none !important;" "allowfullscreen" "webkitallowfullscreen" "mozallowfullscreen"></iframe>



## Get directions via public transport


```python
# Request directions via public transit
now = datetime.now()
directions_result = gmaps.directions(
    "Sydney Town Hall",
    "Parramatta, NSW",
    mode="transit",
    departure_time=now
)
print(json.dumps(directions_result,indent=2))
```

    [
      {
        "bounds": {
          "northeast": {
            "lat": -33.8148186,
            "lng": 151.208821
          },
          "southwest": {
            "lat": -33.89778769999999,
            "lng": 151.0017629
          }
        },
        "copyrights": "Map data \u00a92023 Google",
        "legs": [
          {
            "arrival_time": {
              "text": "11:16\u202fPM",
              "time_zone": "Australia/Sydney",
              "value": 1690291001
            },
            "departure_time": {
              "text": "10:33\u202fPM",
              "time_zone": "Australia/Sydney",
              "value": 1690288420
            },
            "distance": {
              "text": "25.1 km",
              "value": 25071
            },
            "duration": {
              "text": "43 mins",
              "value": 2581
            },
            "end_address": "Parramatta NSW 2150, Australia",
            "end_location": {
              "lat": -33.8148186,
              "lng": 151.0017629
            },
            "start_address": "483 George St, Sydney NSW 2000, Australia",
            "start_location": {
              "lat": -33.8731812,
              "lng": 151.2070056
            },
            "steps": [
              {
                "distance": {
                  "text": "88 m",
                  "value": 88
                },
                "duration": {
                  "text": "1 min",
                  "value": 85
                },
                "end_location": {
                  "lat": -33.8735668,
                  "lng": 151.2069088
                },
                "html_instructions": "Walk to Town Hall",
                "polyline": {
                  "points": "jzvmEyr{y[@@nADCr@@?Ag@"
                },
                "start_location": {
                  "lat": -33.8731812,
                  "lng": 151.2070056
                },
                "steps": [
                  {
                    "distance": {
                      "text": "46 m",
                      "value": 46
                    },
                    "duration": {
                      "text": "1 min",
                      "value": 33
                    },
                    "end_location": {
                      "lat": -33.8735922,
                      "lng": 151.2069691
                    },
                    "html_instructions": "Head <b>south</b> on <b>George St</b>",
                    "polyline": {
                      "points": "jzvmEyr{y[@@nAD"
                    },
                    "start_location": {
                      "lat": -33.8731812,
                      "lng": 151.2070056
                    },
                    "travel_mode": "WALKING"
                  },
                  {
                    "distance": {
                      "text": "24 m",
                      "value": 24
                    },
                    "duration": {
                      "text": "1 min",
                      "value": 16
                    },
                    "end_location": {
                      "lat": -33.8735685,
                      "lng": 151.2067097
                    },
                    "html_instructions": "Turn <b>right</b>",
                    "maneuver": "turn-right",
                    "polyline": {
                      "points": "||vmEqr{y[Cr@"
                    },
                    "start_location": {
                      "lat": -33.8735922,
                      "lng": 151.2069691
                    },
                    "travel_mode": "WALKING"
                  },
                  {
                    "distance": {
                      "text": "1 m",
                      "value": 0
                    },
                    "duration": {
                      "text": "1 min",
                      "value": 3
                    },
                    "end_location": {
                      "lat": -33.8735826,
                      "lng": 151.2067078
                    },
                    "html_instructions": "Take entrance <span class=\"location\">George St</span>",
                    "polyline": {
                      "points": "z|vmE}p{y["
                    },
                    "start_location": {
                      "lat": -33.8735826,
                      "lng": 151.2067078
                    },
                    "travel_mode": "WALKING"
                  },
                  {
                    "distance": {
                      "text": "1 m",
                      "value": 0
                    },
                    "duration": {
                      "text": "1 min",
                      "value": 3
                    },
                    "end_location": {
                      "lat": -33.8735826,
                      "lng": 151.2067078
                    },
                    "polyline": {
                      "points": "z|vmE}p{y["
                    },
                    "start_location": {
                      "lat": -33.8735826,
                      "lng": 151.2067078
                    },
                    "travel_mode": "WALKING"
                  },
                  {
                    "distance": {
                      "text": "1 m",
                      "value": 0
                    },
                    "duration": {
                      "text": "1 min",
                      "value": 2
                    },
                    "end_location": {
                      "lat": -33.8735826,
                      "lng": 151.2067078
                    },
                    "polyline": {
                      "points": "z|vmE}p{y["
                    },
                    "start_location": {
                      "lat": -33.8735826,
                      "lng": 151.2067078
                    },
                    "travel_mode": "WALKING"
                  },
                  {
                    "distance": {
                      "text": "18 m",
                      "value": 18
                    },
                    "duration": {
                      "text": "1 min",
                      "value": 28
                    },
                    "end_location": {
                      "lat": -33.8735668,
                      "lng": 151.2069088
                    },
                    "polyline": {
                      "points": "z|vmE}p{y[Ag@"
                    },
                    "start_location": {
                      "lat": -33.8735826,
                      "lng": 151.2067078
                    },
                    "travel_mode": "WALKING"
                  }
                ],
                "travel_mode": "WALKING"
              },
              {
                "distance": {
                  "text": "24.5 km",
                  "value": 24491
                },
                "duration": {
                  "text": "34 mins",
                  "value": 2040
                },
                "end_location": {
                  "lat": -33.8175076,
                  "lng": 151.0050435
                },
                "html_instructions": "Train towards Penrith Via Central",
                "polyline": {
                  "points": "x|vmEer{y[?Ej@?fCRpCt@\\DN?TCRCZIRGTIf@]b@_@b@e@LQLWLWLULUl@m@t@i@x@c@z@]d@MVI\\I~AQn@CF@hAN@?L@H?H?^Dt@LZDHBB?H@LBZF~BZl@JH@@@TDhARd@JZH`@JB@LBd@NdA^PFPHb@TTLhAp@p@f@`DvEt@d@Z^PR^j@zAvBz@nAf@n@r@zALd@n@dAT^`@l@V\\j@n@PP@@RRNNVRTRZV~B`Bv@j@JHZXJHJHt@p@p@j@t@n@r@n@^^PNJL\\`@X\\JNVZf@l@^l@Zd@BFFJXf@p@nAf@bAP`@Zr@b@hAXr@`@fAXv@Z~@Rp@Pp@Lb@FZJd@RbAFf@Jl@Jn@Ff@Jh@Hj@FZLj@Ld@Pr@Pt@FZ^bBTbAJ`@Lh@Lj@FXNl@BJH\\XlANr@FVPl@Rr@h@pB\\zAV`ALl@Lf@HZLl@R~@FR?BDNBL^|APr@@FZvAJl@NbA\\~C@DFl@F`A@J@VBnA?Z@l@?j@BnA?xA?P?~A?^Av@Cx@G~ACz@Al@?xA?JAXAr@CVCXCTEZKt@Qt@Q|@GRGTK^[`AQf@Sd@IRWh@KTc@x@k@`AYj@OXQ^Sd@KTITKZCFCJ[jAI^ERAFCHENg@pBCHEROd@]nAELa@tAYhAQx@ABCPQdAIh@Ip@CJAJCj@ANAJ?RCf@?JAxA?N@z@?d@B`@?JD~@FrABV@`@NtB@D?DXxC@NNvADf@Db@D|@Bh@A|@AdAGdAKtAMbAc@bCUfAk@bCWvAEXGr@Gp@ALIbAGjA?jABzA@PNbBd@dELrANlBH`B?D?pACjBKjBIfAWdCGt@KdASnBQdAMp@S~@_@vAEPOn@Qt@YpAIb@WrAGVGTc@jBCJc@hBGTI^A@Sx@ABOl@g@rBCLKj@q@|CW`AELa@bB_@rAg@xAOd@ITSl@St@Mf@K`@Ox@QbAMt@QlAKr@K`AGv@Ez@EdBA`AC|@?LA|@Ct@Ap@ATAnAA\\CnACdAAd@Ad@C\\?@Cf@?BEh@Gz@K~@Il@Gd@It@G\\CRId@I^I`@U|@Qx@EPUt@?@Mt@IZGTOn@ABSt@]nA_@rAMh@Oj@Ql@Sj@Mb@Oj@Oj@Ol@Sv@On@K`@On@Ov@Ib@ERADAHCHId@Mn@Kh@On@Mn@Qx@Ov@St@S|@St@YhAQr@Kb@CJIZA?GVGRWx@Sn@M`@KXITABCFKXSf@MXMX_@z@Sd@S`@cAxBc@`AOZEJ{@lBSb@m@tA]p@s@hAa@n@S\\MTOTEJaAbBA@MRGLYh@Ub@S\\k@bAi@v@m@z@eAvAKLcBzBkBdCa@j@u@nAOVCDQVWf@_@v@g@pAWt@Of@W~@Sv@Ml@I`@WnAc@|BWpAQdAWdBWrBGb@OxACXCVCRYnC?B_@rDObBMz@AJOt@Mj@IZUt@AFKXYt@_@z@m@nAEJGJWb@S\\SXUXa@d@]^GF_@`@STUVk@h@w@r@_Ax@q@j@a@^ILQNi@l@c@n@W^GJm@dAs@pAi@~@IJk@z@WZ[\\CD[\\[\\_@`@STIJYZKNKLi@n@e@n@UZ_@h@[d@Ub@OXUd@Yn@Sj@Yx@Ut@Wv@]~@Wn@e@|@k@fA[l@A@OZOVO^EL]v@_@dAUn@EL[x@i@zA[~@?@Sp@IZMh@Ol@Id@S~@ETG\\CJKf@CJEPGPCNOt@EPKf@Ov@Or@S`AI\\?BWjAS`AUjASbAQv@In@CNG\\Gl@Gl@Gp@Cf@Ep@Ab@AZAn@A|@?l@@d@@d@@j@B`@?FBf@?@Dj@JdAJfALlAJxABp@Fz@B|@Dt@@R@R@TBt@B|@?r@?~@At@C|@Ad@Er@Ej@Eh@Gv@Gl@y@lHUfB[|AGb@EXEf@Gd@Gp@g@rESdDGtAGfCE|A?b@CnB?hB@jC@rABdCBlAHxAHhANxALfARtAHXHf@Jr@L~@JnA@T@TBt@@z@Av@CbBEx@EnACfAA~BB|CAfDAtA?~AGrG?DAlADnAAdBAhAKxBC\\Kz@It@Kx@GZGd@Qt@Ol@Mj@ADSl@Up@K\\A@Sl@Wn@O^Wf@OVOVU^QTOTW^W^UX]`@YZUT[XOPWRQNWPMJSLMJMHUNkEnC_Aj@yBrAyH|EiAp@eEhCIFsD~BaHhE[ROJu@d@A?[R}@h@_BbAeAp@SLiAt@g@\\EBqBvAIFgAv@{@t@[X[X[XONCB]\\[ZeBhBIJs@r@EFq@p@uAxAEB_A`AYXMHWPGFq@t@e@f@UVUNSPEDQLIHa@ZWRIHEFe@f@ST]\\WXa@`@MLILEFk@h@i@n@u@v@cAfAq@t@w@v@i@f@c@d@_@\\OHIF]\\cAdAyD|DaCjCyAbBe@l@o@~@y@lAoB`DoAhCkAxCiAvB]p@ELYh@ABc@n@KPEVaDbGg@`AOTwB~DaCpEMVMR?@qArB_CpEw@zAEHy@~AABEFiAtB_@n@cBdDq@|@EHCDY`@ML_@b@ABc@b@e@b@o@d@q@b@SJ_@PIBMDgCfAq@Tq@RmA^aDz@[FUDSDK@O?m@FgBHyBDiDMeDYcE{@mCUsBCS@aABaBFkB`@oAd@a@V]RQLgAr@oA`Aa@\\OLMJa@\\{@`A]d@k@dAGLRN"
                },
                "start_location": {
                  "lat": -33.8735668,
                  "lng": 151.2069088
                },
                "transit_details": {
                  "arrival_stop": {
                    "location": {
                      "lat": -33.8175076,
                      "lng": 151.0050435
                    },
                    "name": "Parramatta"
                  },
                  "arrival_time": {
                    "text": "11:10\u202fPM",
                    "time_zone": "Australia/Sydney",
                    "value": 1690290600
                  },
                  "departure_stop": {
                    "location": {
                      "lat": -33.8735668,
                      "lng": 151.2069088
                    },
                    "name": "Town Hall"
                  },
                  "departure_time": {
                    "text": "10:36\u202fPM",
                    "time_zone": "Australia/Sydney",
                    "value": 1690288560
                  },
                  "headsign": "Penrith Via Central",
                  "line": {
                    "agencies": [
                      {
                        "name": "Sydney Trains",
                        "url": "https://transportnsw.info/"
                      }
                    ],
                    "color": "#f2991b",
                    "name": "North Shore & Western Line",
                    "short_name": "T1",
                    "text_color": "#ffffff",
                    "vehicle": {
                      "icon": "//maps.gstatic.com/mapfiles/transit/iw2/6/rail2.png",
                      "local_icon": "//maps.gstatic.com/mapfiles/transit/iw2/6/au-sydney-train.png",
                      "name": "Train",
                      "type": "HEAVY_RAIL"
                    }
                  },
                  "num_stops": 7
                },
                "travel_mode": "TRANSIT"
              },
              {
                "distance": {
                  "text": "0.5 km",
                  "value": 492
                },
                "duration": {
                  "text": "7 mins",
                  "value": 401
                },
                "end_location": {
                  "lat": -33.8148186,
                  "lng": 151.0017629
                },
                "html_instructions": "Walk to Parramatta NSW 2150, Australia",
                "polyline": {
                  "points": "l~kmEodtx[aB\\ECOj@ETCNAHGXGZCLAD?@CLG^CRERALAFAJKFMFSHGBEBGBMDq@JmAPE@EBKHMFIFOBG@E?E?IACNAHIh@Ip@YpB"
                },
                "start_location": {
                  "lat": -33.8175076,
                  "lng": 151.0050435
                },
                "steps": [
                  {
                    "distance": {
                      "text": "1 m",
                      "value": 0
                    },
                    "duration": {
                      "text": "1 min",
                      "value": 60
                    },
                    "end_location": {
                      "lat": -33.8175076,
                      "lng": 151.0050435
                    },
                    "polyline": {
                      "points": "l~kmEodtx["
                    },
                    "start_location": {
                      "lat": -33.8175076,
                      "lng": 151.0050435
                    },
                    "travel_mode": "WALKING"
                  },
                  {
                    "distance": {
                      "text": "1 m",
                      "value": 0
                    },
                    "duration": {
                      "text": "1 min",
                      "value": 12
                    },
                    "end_location": {
                      "lat": -33.8175076,
                      "lng": 151.0050435
                    },
                    "polyline": {
                      "points": "l~kmEodtx["
                    },
                    "start_location": {
                      "lat": -33.8175076,
                      "lng": 151.0050435
                    },
                    "travel_mode": "WALKING"
                  },
                  {
                    "distance": {
                      "text": "1 m",
                      "value": 0
                    },
                    "duration": {
                      "text": "1 min",
                      "value": 6
                    },
                    "end_location": {
                      "lat": -33.8175076,
                      "lng": 151.0050435
                    },
                    "polyline": {
                      "points": "l~kmEodtx["
                    },
                    "start_location": {
                      "lat": -33.8175076,
                      "lng": 151.0050435
                    },
                    "travel_mode": "WALKING"
                  },
                  {
                    "distance": {
                      "text": "1 m",
                      "value": 0
                    },
                    "duration": {
                      "text": "1 min",
                      "value": 22
                    },
                    "end_location": {
                      "lat": -33.8175076,
                      "lng": 151.0050435
                    },
                    "polyline": {
                      "points": "l~kmEodtx["
                    },
                    "start_location": {
                      "lat": -33.8175076,
                      "lng": 151.0050435
                    },
                    "travel_mode": "WALKING"
                  },
                  {
                    "distance": {
                      "text": "55 m",
                      "value": 55
                    },
                    "duration": {
                      "text": "1 min",
                      "value": 2
                    },
                    "end_location": {
                      "lat": -33.8170243,
                      "lng": 151.0048913
                    },
                    "html_instructions": "Take exit <span class=\"location\">Darcy St</span>",
                    "polyline": {
                      "points": "l~kmEodtx[aB\\"
                    },
                    "start_location": {
                      "lat": -33.8175076,
                      "lng": 151.0050435
                    },
                    "travel_mode": "WALKING"
                  },
                  {
                    "distance": {
                      "text": "0.1 km",
                      "value": 143
                    },
                    "duration": {
                      "text": "2 mins",
                      "value": 94
                    },
                    "end_location": {
                      "lat": -33.8166036,
                      "lng": 151.0034729
                    },
                    "html_instructions": "Head <b>northwest</b> on <b>Darcy St</b>/<wbr/><b>Parramatta Sq</b> toward <b>Church St</b>",
                    "polyline": {
                      "points": "d{kmEuctx[Oj@ETCNAHGXGZCLAD?@CLG^CRERALAFAJ"
                    },
                    "start_location": {
                      "lat": -33.8169924,
                      "lng": 151.0049083
                    },
                    "travel_mode": "WALKING"
                  },
                  {
                    "distance": {
                      "text": "0.2 km",
                      "value": 180
                    },
                    "duration": {
                      "text": "2 mins",
                      "value": 128
                    },
                    "end_location": {
                      "lat": -33.815075,
                      "lng": 151.0029162
                    },
                    "html_instructions": "Turn <b>right</b> onto <b>Church St</b>",
                    "maneuver": "turn-right",
                    "polyline": {
                      "points": "vxkmEuzsx[KFMFSHGBEBGBMDq@JmAPE@EBKHMFIFOBG@E?E?IA"
                    },
                    "start_location": {
                      "lat": -33.8166036,
                      "lng": 151.0034729
                    },
                    "travel_mode": "WALKING"
                  },
                  {
                    "distance": {
                      "text": "0.1 km",
                      "value": 114
                    },
                    "duration": {
                      "text": "1 min",
                      "value": 77
                    },
                    "end_location": {
                      "lat": -33.8148186,
                      "lng": 151.0017629
                    },
                    "html_instructions": "Turn <b>left</b> onto <b>Macquarie St</b>",
                    "maneuver": "turn-left",
                    "polyline": {
                      "points": "fokmEgwsx[CNAHIh@Ip@YpB"
                    },
                    "start_location": {
                      "lat": -33.815075,
                      "lng": 151.0029162
                    },
                    "travel_mode": "WALKING"
                  }
                ],
                "travel_mode": "WALKING"
              }
            ],
            "traffic_speed_entry": [],
            "via_waypoint": []
          }
        ],
        "overview_polyline": {
          "points": "jzvmEyr{y[pAFAr@Am@j@?fCRpCt@\\Dd@Cn@Mh@QjA}@p@w@v@{Al@m@t@i@x@c@z@]|@W\\I~AQn@CF@jAN`AFrG`AnATnB^|@T|Bt@b@Px@b@hAp@p@f@`DvEt@d@l@r@zBbDbB~Br@zALd@dAdBx@jArAvAxAnAvDlCf@b@~BpBzCnCnA|A~@hAz@rAvAjCx@dB~@|BpBrFz@bD^hBRtAh@lDt@`D~BjKv@dD|@lDlB|Hx@pD`BfHZpB^dDRrCDdEBzGAvAKxCEhB?dBK~BIp@]jBm@fCm@hB]x@sBzDoAjCe@nAo@jCs@rCw@rCg@bBk@bCa@dCStBGvA?pDHrC\\bG\\tDT~BJ`B@fBIjCYxCy@jEk@bCWvAMlAI~@QnCBfDPtBd@dE\\`EHfBC|DUrD_@zD_@tD_@vBiAxEmA~FoBfImA~EOx@iA~Eg@pBgAlDaA~CYhAa@|B_@bCWtBMrBGfDI~DSvKGjAMdBUlB]lCs@dDm@bCo@vCsA|EcAnD}@hD_AxD_A~E{@bEkA`FwAtFeAdD[z@oAvCoCbGeBxDkAfCuAxBw@tAyAfCoBnDiDxEoE`GwAzB}@|AgAhCg@|Ak@vBWnAeBdJo@xEW|Be@tE_@vD]~CQ`AWfA}@lC{AbDk@`Ai@r@gBnBmCjCqBdBk@l@{@|@{@nAsCbFmAbBoD~DgB|Bu@dAq@hAe@~@m@zAeBfF}@lBgAtBq@tAyAzDgBbFk@xBs@hD}@fEgBtIoAhG]jCYxDGlDDdDFrAj@fGZdGJrBFrB?rBMlE[zDoAtKc@`C[xCg@rESdDO|EE`CCxEJrLRbDNxA`@|CR`AXrBLdBDjA?rBStH@|GC|FIfMDnAAdBAhAKxBOxAUnBO`Aa@bBy@pCy@|Bg@fA_@n@oAjBm@x@w@|@yAtAkAz@{XdQiW`P_GrDkCdB{B~AcClBcB|AgFnFmEpEg@b@_@XwA|Ak@f@u@n@iA`AoBtBoG~GwDzDo@f@IF]\\}FbG{EnFuAlBiDnFoAhCkAxCgBhD_@v@e@r@Qh@iEdIwG~LMTqArB_CpE}@dB{@bBsErIuAnBsAxAuAhAeAn@_EbBcBh@oFzAqATO?m@FaFNiDMeDYcE{@mCUgCAcDJkB`@oAd@_Aj@yA`AqB~A]Xa@\\{@`A]d@k@dAGLRNaB\\ECU`AUnA]xBCRYNi@TsCd@i@\\c@DIACNKr@c@bD"
        },
        "summary": "",
        "warnings": [
          "Walking directions are in beta. Use caution \u2013 This route may be missing sidewalks or pedestrian paths."
        ],
        "waypoint_order": []
      }
    ]



```python
res = directions_result[0]
print('distance:',res['legs'][0]['distance']['text'])
print('duration:',res['legs'][0]['duration']['text'])
print('start:',res['legs'][0]['start_location'])
print('end:',res['legs'][0]['end_location'])
```

    distance: 25.1 km
    duration: 43 mins
    start: {'lat': -33.8731812, 'lng': 151.2070056}
    end: {'lat': -33.8148186, 'lng': 151.0017629}



```python
# https://qiita.com/shin1007/items/8f31188b9ac83be9e738
def decode_polyline(enc: str):
    """
    Parameters
    ----------
    enc : str
        encoded string of polyline, which can be aquired via Google Maps API.

    Returns
    -------
    result : list
        each element in `result` contains pair of latitude and longitude.
    """
    if enc == None or enc == '':
        return [[0, 0]]

    result = []
    polyline_chars = list(enc.encode())
    current_latitude = 0
    current_longitude = 0
    try:
        index = 0
        while index < len(polyline_chars):
            # calculate next latitude
            total = 0
            shifter = 0

            while True:
                next5bits = int(polyline_chars[index]) - 63
                index += 1
                total |= (next5bits & 31) << shifter
                shifter += 5
                if not(next5bits >= 32 and index < len(polyline_chars)):
                    break

            if (index >= len(polyline_chars)):
                break

            if((total & 1) == 1):
                current_latitude += ~(total >> 1)
            else:
                current_latitude += (total >> 1)

            # calculate next longitude
            total = 0
            shifter = 0
            while True:
                next5bits = int(polyline_chars[index]) - 63
                index += 1
                total |= (next5bits & 31) << shifter
                shifter += 5
                if not(next5bits >= 32 and index < len(polyline_chars)):
                    break

            if (index >= len(polyline_chars) and next >= 32):
                break

            if((total & 1) == 1):
                current_longitude += ~(total >> 1)
            else:
                current_longitude += (total >> 1)

            # add to return value
            pair = [current_latitude / 100000, current_longitude / 100000]
            result.append(pair)

    except:
        pass
    return result
```


```python
color_map = {
    'WALKING': '#0000ff',
    'TRANSIT': '#ff0000',
}
```


```python
steps = []
for i,s in enumerate(directions_result[0]['legs'][0]['steps']):
    print(i,s['travel_mode'])
    steps.append({
        'polyline': decode_polyline(s['polyline']['points']),
        'travel_mode': s['travel_mode'],
        'duration': s['duration']['text'],
        'duration_sec': s['duration']['value']
    })
start,end = res['legs'][0]['start_location'],res['legs'][0]['end_location']
lat,lng = (start['lat']+end['lat'])/2,(start['lng']+end['lng'])/2
map = folium.Map(location=[lat,lng], zoom_start=12)
folium.Marker(location=[start['lat'],start['lng']], popup=res['legs'][0]['start_address']).add_to(map)
folium.Marker(location=[end['lat'],end['lng']], popup=res['legs'][0]['end_address']).add_to(map)

for step in steps:
    folium.vector_layers.PolyLine(
        locations=step['polyline'], popup=f'{step["travel_mode"]} ({step["duration"]})', color=color_map[step['travel_mode']]
    ).add_to(map)
fig = Figure(width=800, height=600)
fig.add_child(map)
```

    0 WALKING
    1 TRANSIT
    2 WALKING





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
                #map_14b774d42b52df4d0da4b851913953f2 {
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


            &lt;div class=&quot;folium-map&quot; id=&quot;map_14b774d42b52df4d0da4b851913953f2&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_14b774d42b52df4d0da4b851913953f2 = L.map(
                &quot;map_14b774d42b52df4d0da4b851913953f2&quot;,
                {
                    center: [-33.8439999, 151.10438425],
                    crs: L.CRS.EPSG3857,
                    zoom: 12,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );





            var tile_layer_ad7c5c634883e1a32b5ad22a7d105f75 = L.tileLayer(
                &quot;https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;Data by \u0026copy; \u003ca target=\&quot;_blank\&quot; href=\&quot;http://openstreetmap.org\&quot;\u003eOpenStreetMap\u003c/a\u003e, under \u003ca target=\&quot;_blank\&quot; href=\&quot;http://www.openstreetmap.org/copyright\&quot;\u003eODbL\u003c/a\u003e.&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 18, &quot;maxZoom&quot;: 18, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            ).addTo(map_14b774d42b52df4d0da4b851913953f2);


            var marker_04700664404ee3e45e092b4d7060e41b = L.marker(
                [-33.8731812, 151.2070056],
                {}
            ).addTo(map_14b774d42b52df4d0da4b851913953f2);


        var popup_49c8dd60816f0099a02424e6417efc49 = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_08b3b634e53e8f134d9825c855f7ad93 = $(`&lt;div id=&quot;html_08b3b634e53e8f134d9825c855f7ad93&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;483 George St, Sydney NSW 2000, Australia&lt;/div&gt;`)[0];
                popup_49c8dd60816f0099a02424e6417efc49.setContent(html_08b3b634e53e8f134d9825c855f7ad93);



        marker_04700664404ee3e45e092b4d7060e41b.bindPopup(popup_49c8dd60816f0099a02424e6417efc49)
        ;




            var marker_1f9c408583145fda844e3b741a7e3069 = L.marker(
                [-33.8148186, 151.0017629],
                {}
            ).addTo(map_14b774d42b52df4d0da4b851913953f2);


        var popup_66ca2062fde71d6e1d8b2058ae7b450f = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_88cb08643df6b5f637261af5c2cdf19f = $(`&lt;div id=&quot;html_88cb08643df6b5f637261af5c2cdf19f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;Parramatta NSW 2150, Australia&lt;/div&gt;`)[0];
                popup_66ca2062fde71d6e1d8b2058ae7b450f.setContent(html_88cb08643df6b5f637261af5c2cdf19f);



        marker_1f9c408583145fda844e3b741a7e3069.bindPopup(popup_66ca2062fde71d6e1d8b2058ae7b450f)
        ;




            var poly_line_9420d64fc0eb78f466312054c435c29d = L.polyline(
                [[-33.87318, 151.20701], [-33.87319, 151.207], [-33.87359, 151.20697], [-33.87357, 151.20671], [-33.87358, 151.20671]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#0000ff&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;#0000ff&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_14b774d42b52df4d0da4b851913953f2);


        var popup_1a4a466d0858db409fea9ca426e299de = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_e429f5a1718da26b9a0d2ec30e106f1e = $(`&lt;div id=&quot;html_e429f5a1718da26b9a0d2ec30e106f1e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;WALKING (1 min)&lt;/div&gt;`)[0];
                popup_1a4a466d0858db409fea9ca426e299de.setContent(html_e429f5a1718da26b9a0d2ec30e106f1e);



        poly_line_9420d64fc0eb78f466312054c435c29d.bindPopup(popup_1a4a466d0858db409fea9ca426e299de)
        ;




            var poly_line_bf10cd2779b943249f29cd6067d1bc50 = L.polyline(
                [[-33.87357, 151.20691], [-33.87357, 151.20694], [-33.87379, 151.20694], [-33.87447, 151.20684], [-33.8752, 151.20657], [-33.87535, 151.20654], [-33.87543, 151.20654], [-33.87554, 151.20656], [-33.87564, 151.20658], [-33.87578, 151.20663], [-33.87588, 151.20667], [-33.87599, 151.20672], [-33.87619, 151.20687], [-33.87637, 151.20703], [-33.87655, 151.20722], [-33.87662, 151.20731], [-33.87669, 151.20743], [-33.87676, 151.20755], [-33.87683, 151.20766], [-33.8769, 151.20777], [-33.87713, 151.208], [-33.8774, 151.20821], [-33.87769, 151.20839], [-33.87799, 151.20854], [-33.87818, 151.20861], [-33.8783, 151.20866], [-33.87845, 151.20871], [-33.87893, 151.2088], [-33.87917, 151.20882], [-33.87921, 151.20881], [-33.87958, 151.20873], [-33.87959, 151.20873], [-33.87966, 151.20872], [-33.87971, 151.20872], [-33.87976, 151.20872], [-33.87992, 151.20869], [-33.88019, 151.20862], [-33.88033, 151.20859], [-33.88038, 151.20857], [-33.8804, 151.20857], [-33.88045, 151.20856], [-33.88052, 151.20854], [-33.88066, 151.2085], [-33.8813, 151.20836], [-33.88153, 151.2083], [-33.88158, 151.20829], [-33.88159, 151.20828], [-33.8817, 151.20825], [-33.88207, 151.20815], [-33.88226, 151.20809], [-33.8824, 151.20804], [-33.88257, 151.20798], [-33.88259, 151.20797], [-33.88266, 151.20795], [-33.88285, 151.20787], [-33.8832, 151.20771], [-33.88329, 151.20767], [-33.88338, 151.20762], [-33.88356, 151.20751], [-33.88367, 151.20744], [-33.88404, 151.20719], [-33.88429, 151.20699], [-33.8851, 151.20591], [-33.88537, 151.20572], [-33.88551, 151.20556], [-33.8856, 151.20546], [-33.88576, 151.20524], [-33.88622, 151.20464], [-33.88652, 151.20424], [-33.88672, 151.204], [-33.88698, 151.20354], [-33.88705, 151.20335], [-33.88729, 151.203], [-33.8874, 151.20284], [-33.88757, 151.20261], [-33.88769, 151.20246], [-33.88791, 151.20222], [-33.888, 151.20213], [-33.88801, 151.20212], [-33.88811, 151.20202], [-33.88819, 151.20194], [-33.88831, 151.20184], [-33.88842, 151.20174], [-33.88856, 151.20162], [-33.8892, 151.20113], [-33.88948, 151.20091], [-33.88954, 151.20086], [-33.88968, 151.20073], [-33.88974, 151.20068], [-33.8898, 151.20063], [-33.89007, 151.20038], [-33.89032, 151.20016], [-33.89059, 151.19992], [-33.89085, 151.19968], [-33.89101, 151.19952], [-33.8911, 151.19944], [-33.89116, 151.19937], [-33.89131, 151.1992], [-33.89144, 151.19905], [-33.8915, 151.19897], [-33.89162, 151.19883], [-33.89182, 151.1986], [-33.89198, 151.19837], [-33.89212, 151.19818], [-33.89214, 151.19814], [-33.89218, 151.19808], [-33.89231, 151.19788], [-33.89256, 151.19748], [-33.89276, 151.19714], [-33.89285, 151.19697], [-33.89299, 151.19671], [-33.89317, 151.19634], [-33.8933, 151.19608], [-33.89347, 151.19572], [-33.8936, 151.19544], [-33.89374, 151.19512], [-33.89384, 151.19487], [-33.89393, 151.19462], [-33.894, 151.19444], [-33.89404, 151.1943], [-33.8941, 151.19411], [-33.8942, 151.19377], [-33.89424, 151.19357], [-33.8943, 151.19334], [-33.89436, 151.1931], [-33.8944, 151.1929], [-33.89446, 151.19269], [-33.89451, 151.19247], [-33.89455, 151.19233], [-33.89462, 151.19211], [-33.89469, 151.19192], [-33.89478, 151.19166], [-33.89487, 151.19139], [-33.89491, 151.19125], [-33.89507, 151.19075], [-33.89518, 151.19041], [-33.89524, 151.19024], [-33.89531, 151.19003], [-33.89538, 151.18981], [-33.89542, 151.18968], [-33.8955, 151.18945], [-33.89552, 151.18939], [-33.89557, 151.18924], [-33.8957, 151.18885], [-33.89578, 151.18859], [-33.89582, 151.18847], [-33.89591, 151.18824], [-33.89601, 151.18798], [-33.89622, 151.18741], [-33.89637, 151.18695], [-33.89649, 151.18662], [-33.89656, 151.18639], [-33.89663, 151.18619], [-33.89668, 151.18605], [-33.89675, 151.18582], [-33.89685, 151.1855], [-33.89689, 151.1854], [-33.89689, 151.18538], [-33.89692, 151.1853], [-33.89694, 151.18523], [-33.8971, 151.18476], [-33.89719, 151.1845], [-33.8972, 151.18446], [-33.89734, 151.18402], [-33.8974, 151.18379], [-33.89748, 151.18345], [-33.89763, 151.18265], [-33.89764, 151.18262], [-33.89768, 151.18239], [-33.89772, 151.18206], [-33.89773, 151.182], [-33.89774, 151.18188], [-33.89776, 151.18148], [-33.89776, 151.18134], [-33.89777, 151.18111], [-33.89777, 151.18089], [-33.89779, 151.18049], [-33.89779, 151.18004], [-33.89779, 151.17995], [-33.89779, 151.17947], [-33.89779, 151.17931], [-33.89778, 151.17903], [-33.89776, 151.17874], [-33.89772, 151.17826], [-33.8977, 151.17796], [-33.89769, 151.17773], [-33.89769, 151.17728], [-33.89769, 151.17722], [-33.89768, 151.17709], [-33.89767, 151.17683], [-33.89765, 151.17671], [-33.89763, 151.17658], [-33.89761, 151.17647], [-33.89758, 151.17633], [-33.89752, 151.17606], [-33.89743, 151.17579], [-33.89734, 151.17548], [-33.8973, 151.17538], [-33.89726, 151.17527], [-33.8972, 151.17511], [-33.89706, 151.17478], [-33.89697, 151.17458], [-33.89687, 151.17439], [-33.89682, 151.17429], [-33.8967, 151.17408], [-33.89664, 151.17397], [-33.89646, 151.17368], [-33.89624, 151.17335], [-33.89611, 151.17313], [-33.89603, 151.173], [-33.89594, 151.17284], [-33.89584, 151.17265], [-33.89578, 151.17254], [-33.89573, 151.17243], [-33.89567, 151.17229], [-33.89565, 151.17225], [-33.89563, 151.17219], [-33.89549, 151.17181], [-33.89544, 151.17165], [-33.89541, 151.17155], [-33.8954, 151.17151], [-33.89538, 151.17146], [-33.89535, 151.17138], [-33.89515, 151.17081], [-33.89513, 151.17076], [-33.8951, 151.17066], [-33.89502, 151.17047], [-33.89487, 151.17007], [-33.89484, 151.17], [-33.89467, 151.16957], [-33.89454, 151.1692], [-33.89445, 151.16891], [-33.89444, 151.16889], [-33.89442, 151.1688], [-33.89433, 151.16845], [-33.89428, 151.16824], [-33.89423, 151.16799], [-33.89421, 151.16793], [-33.8942, 151.16787], [-33.89418, 151.16765], [-33.89417, 151.16757], [-33.89416, 151.16751], [-33.89416, 151.16741], [-33.89414, 151.16721], [-33.89414, 151.16715], [-33.89413, 151.1667], [-33.89413, 151.16662], [-33.89414, 151.16632], [-33.89414, 151.16613], [-33.89416, 151.16596], [-33.89416, 151.1659], [-33.89419, 151.16558], [-33.89423, 151.16516], [-33.89425, 151.16504], [-33.89426, 151.16487], [-33.89434, 151.16428], [-33.89435, 151.16425], [-33.89435, 151.16422], [-33.89448, 151.16345], [-33.89449, 151.16337], [-33.89457, 151.16293], [-33.8946, 151.16273], [-33.89463, 151.16255], [-33.89466, 151.16224], [-33.89468, 151.16203], [-33.89467, 151.16172], [-33.89466, 151.16137], [-33.89462, 151.16102], [-33.89456, 151.16059], [-33.89449, 151.16025], [-33.89431, 151.15959], [-33.8942, 151.15923], [-33.89398, 151.15857], [-33.89386, 151.15813], [-33.89383, 151.158], [-33.89379, 151.15774], [-33.89375, 151.15749], [-33.89374, 151.15742], [-33.89369, 151.15708], [-33.89365, 151.1567], [-33.89365, 151.15632], [-33.89367, 151.15586], [-33.89368, 151.15577], [-33.89376, 151.15527], [-33.89395, 151.15428], [-33.89402, 151.15386], [-33.8941, 151.15331], [-33.89415, 151.15282], [-33.89415, 151.15279], [-33.89415, 151.15238], [-33.89413, 151.15184], [-33.89407, 151.1513], [-33.89402, 151.15094], [-33.8939, 151.15027], [-33.89386, 151.15], [-33.8938, 151.14965], [-33.8937, 151.14909], [-33.89361, 151.14874], [-33.89354, 151.14849], [-33.89344, 151.14817], [-33.89328, 151.14773], [-33.89325, 151.14764], [-33.89317, 151.1474], [-33.89308, 151.14713], [-33.89295, 151.14672], [-33.8929, 151.14654], [-33.89278, 151.14612], [-33.89274, 151.146], [-33.8927, 151.14589], [-33.89252, 151.14535], [-33.8925, 151.14529], [-33.89232, 151.14476], [-33.89228, 151.14465], [-33.89223, 151.14449], [-33.89222, 151.14448], [-33.89212, 151.14419], [-33.89211, 151.14417], [-33.89203, 151.14394], [-33.89183, 151.14336], [-33.89181, 151.14329], [-33.89175, 151.14307], [-33.8915, 151.14228], [-33.89138, 151.14195], [-33.89135, 151.14188], [-33.89118, 151.14138], [-33.89102, 151.14096], [-33.89082, 151.14051], [-33.89074, 151.14032], [-33.89069, 151.14021], [-33.89059, 151.13998], [-33.89049, 151.13971], [-33.89042, 151.13951], [-33.89036, 151.13934], [-33.89028, 151.13905], [-33.89019, 151.13871], [-33.89012, 151.13844], [-33.89003, 151.13805], [-33.88997, 151.13779], [-33.88991, 151.13746], [-33.88987, 151.13718], [-33.88984, 151.13688], [-33.88981, 151.13637], [-33.8898, 151.13604], [-33.88978, 151.13573], [-33.88978, 151.13566], [-33.88977, 151.13535], [-33.88975, 151.13508], [-33.88974, 151.13483], [-33.88973, 151.13472], [-33.88972, 151.13432], [-33.88971, 151.13417], [-33.88969, 151.13377], [-33.88967, 151.13342], [-33.88966, 151.13323], [-33.88965, 151.13304], [-33.88963, 151.13289], [-33.88963, 151.13288], [-33.88961, 151.13268], [-33.88961, 151.13266], [-33.88958, 151.13245], [-33.88954, 151.13215], [-33.88948, 151.13183], [-33.88943, 151.1316], [-33.88939, 151.13141], [-33.88934, 151.13114], [-33.8893, 151.13099], [-33.88928, 151.13089], [-33.88923, 151.1307], [-33.88918, 151.13054], [-33.88913, 151.13037], [-33.88902, 151.13006], [-33.88893, 151.12977], [-33.8889, 151.12968], [-33.88879, 151.12941], [-33.88879, 151.1294], [-33.88872, 151.12913], [-33.88867, 151.12899], [-33.88863, 151.12888], [-33.88855, 151.12864], [-33.88854, 151.12862], [-33.88844, 151.12835], [-33.88829, 151.12795], [-33.88813, 151.12753], [-33.88806, 151.12732], [-33.88798, 151.1271], [-33.88789, 151.12687], [-33.88779, 151.12665], [-33.88772, 151.12647], [-33.88764, 151.12625], [-33.88756, 151.12603], [-33.88748, 151.1258], [-33.88738, 151.12552], [-33.8873, 151.12528], [-33.88724, 151.12511], [-33.88716, 151.12487], [-33.88708, 151.12459], [-33.88703, 151.12441], [-33.887, 151.12431], [-33.88699, 151.12428], [-33.88698, 151.12423], [-33.88696, 151.12418], [-33.88691, 151.12399], [-33.88684, 151.12375], [-33.88678, 151.12354], [-33.8867, 151.1233], [-33.88663, 151.12306], [-33.88654, 151.12277], [-33.88646, 151.12249], [-33.88636, 151.12222], [-33.88626, 151.12191], [-33.88616, 151.12164], [-33.88603, 151.12127], [-33.88594, 151.12101], [-33.88588, 151.12083], [-33.88586, 151.12077], [-33.88581, 151.12063], [-33.8858, 151.12063], [-33.88576, 151.12051], [-33.88572, 151.12041], [-33.8856, 151.12012], [-33.8855, 151.11988], [-33.88543, 151.11971], [-33.88537, 151.11958], [-33.88532, 151.11947], [-33.88531, 151.11945], [-33.88529, 151.11941], [-33.88523, 151.11928], [-33.88513, 151.11908], [-33.88506, 151.11895], [-33.88499, 151.11882], [-33.88483, 151.11852], [-33.88473, 151.11833], [-33.88463, 151.11816], [-33.88429, 151.11755], [-33.88411, 151.11722], [-33.88403, 151.11708], [-33.884, 151.11702], [-33.8837, 151.11647], [-33.8836, 151.11629], [-33.88337, 151.11586], [-33.88322, 151.11561], [-33.88296, 151.11524], [-33.88279, 151.115], [-33.88269, 151.11485], [-33.88262, 151.11474], [-33.88254, 151.11463], [-33.88251, 151.11457], [-33.88218, 151.11407], [-33.88217, 151.11406], [-33.8821, 151.11396], [-33.88206, 151.11389], [-33.88193, 151.11368], [-33.88182, 151.1135], [-33.88172, 151.11335], [-33.8815, 151.11301], [-33.88129, 151.11273], [-33.88106, 151.11243], [-33.88071, 151.11199], [-33.88065, 151.11192], [-33.88015, 151.1113], [-33.87961, 151.11063], [-33.87944, 151.11041], [-33.87917, 151.11001], [-33.87909, 151.10989], [-33.87907, 151.10986], [-33.87898, 151.10974], [-33.87886, 151.10954], [-33.8787, 151.10926], [-33.8785, 151.10885], [-33.87838, 151.10858], [-33.8783, 151.10838], [-33.87818, 151.10806], [-33.87808, 151.10778], [-33.87801, 151.10755], [-33.87796, 151.10738], [-33.87784, 151.10698], [-33.87766, 151.10635], [-33.87754, 151.10594], [-33.87745, 151.10559], [-33.87733, 151.10508], [-33.87721, 151.1045], [-33.87717, 151.10432], [-33.87709, 151.10387], [-33.87707, 151.10374], [-33.87705, 151.10362], [-33.87703, 151.10352], [-33.8769, 151.1028], [-33.8769, 151.10278], [-33.87674, 151.10188], [-33.87666, 151.10138], [-33.87659, 151.10108], [-33.87658, 151.10102], [-33.8765, 151.10075], [-33.87643, 151.10053], [-33.87638, 151.10039], [-33.87627, 151.10012], [-33.87626, 151.10008], [-33.8762, 151.09995], [-33.87607, 151.09968], [-33.87591, 151.09938], [-33.87568, 151.09898], [-33.87565, 151.09892], [-33.87561, 151.09886], [-33.87549, 151.09868], [-33.87539, 151.09853], [-33.87529, 151.0984], [-33.87518, 151.09827], [-33.87501, 151.09808], [-33.87486, 151.09792], [-33.87482, 151.09788], [-33.87466, 151.09771], [-33.87456, 151.0976], [-33.87445, 151.09748], [-33.87423, 151.09727], [-33.87395, 151.09701], [-33.87363, 151.09672], [-33.87338, 151.0965], [-33.87321, 151.09634], [-33.87316, 151.09627], [-33.87307, 151.09619], [-33.87286, 151.09596], [-33.87268, 151.09572], [-33.87256, 151.09556], [-33.87252, 151.0955], [-33.87229, 151.09515], [-33.87203, 151.09474], [-33.87182, 151.09442], [-33.87177, 151.09436], [-33.87155, 151.09406], [-33.87143, 151.09392], [-33.87129, 151.09377], [-33.87127, 151.09374], [-33.87113, 151.09359], [-33.87099, 151.09344], [-33.87083, 151.09327], [-33.87073, 151.09316], [-33.87068, 151.0931], [-33.87055, 151.09296], [-33.87049, 151.09288], [-33.87043, 151.09281], [-33.87022, 151.09257], [-33.87003, 151.09233], [-33.86992, 151.09219], [-33.86976, 151.09198], [-33.86962, 151.09179], [-33.86951, 151.09161], [-33.86943, 151.09148], [-33.86932, 151.09129], [-33.86919, 151.09105], [-33.86909, 151.09083], [-33.86896, 151.09054], [-33.86885, 151.09027], [-33.86873, 151.08999], [-33.86858, 151.08967], [-33.86846, 151.08943], [-33.86827, 151.08912], [-33.86805, 151.08876], [-33.86791, 151.08853], [-33.8679, 151.08852], [-33.86782, 151.08838], [-33.86774, 151.08826], [-33.86766, 151.0881], [-33.86763, 151.08803], [-33.86748, 151.08775], [-33.86732, 151.0874], [-33.86721, 151.08716], [-33.86718, 151.08709], [-33.86704, 151.0868], [-33.86683, 151.08634], [-33.86669, 151.08602], [-33.86669, 151.08601], [-33.86659, 151.08576], [-33.86654, 151.08562], [-33.86647, 151.08541], [-33.86639, 151.08518], [-33.86634, 151.08499], [-33.86624, 151.08467], [-33.86621, 151.08456], [-33.86617, 151.08441], [-33.86615, 151.08435], [-33.86609, 151.08415], [-33.86607, 151.08409], [-33.86604, 151.084], [-33.866, 151.08391], [-33.86598, 151.08383], [-33.8659, 151.08356], [-33.86587, 151.08347], [-33.86581, 151.08327], [-33.86573, 151.08299], [-33.86565, 151.08273], [-33.86555, 151.0824], [-33.8655, 151.08225], [-33.8655, 151.08223], [-33.86538, 151.08185], [-33.86528, 151.08152], [-33.86517, 151.08114], [-33.86507, 151.0808], [-33.86498, 151.08052], [-33.86493, 151.08028], [-33.86491, 151.0802], [-33.86487, 151.08005], [-33.86483, 151.07982], [-33.86479, 151.07959], [-33.86475, 151.07934], [-33.86473, 151.07914], [-33.8647, 151.07889], [-33.86469, 151.07871], [-33.86468, 151.07857], [-33.86467, 151.07833], [-33.86466, 151.07802], [-33.86466, 151.07779], [-33.86467, 151.0776], [-33.86468, 151.07741], [-33.86469, 151.07719], [-33.86471, 151.07702], [-33.86471, 151.07698], [-33.86473, 151.07678], [-33.86473, 151.07677], [-33.86476, 151.07655], [-33.86482, 151.0762], [-33.86488, 151.07584], [-33.86495, 151.07545], [-33.86501, 151.075], [-33.86503, 151.07475], [-33.86507, 151.07445], [-33.86509, 151.07414], [-33.86512, 151.07387], [-33.86513, 151.07377], [-33.86514, 151.07367], [-33.86515, 151.07356], [-33.86517, 151.07329], [-33.86519, 151.07298], [-33.86519, 151.07272], [-33.86519, 151.0724], [-33.86518, 151.07213], [-33.86516, 151.07182], [-33.86515, 151.07163], [-33.86512, 151.07137], [-33.86509, 151.07115], [-33.86506, 151.07094], [-33.86502, 151.07066], [-33.86498, 151.07043], [-33.86469, 151.06892], [-33.86458, 151.0684], [-33.86444, 151.06793], [-33.8644, 151.06775], [-33.86437, 151.06762], [-33.86434, 151.06742], [-33.8643, 151.06723], [-33.86426, 151.06698], [-33.86406, 151.06592], [-33.86396, 151.06509], [-33.86392, 151.06466], [-33.86388, 151.06398], [-33.86385, 151.06351], [-33.86385, 151.06333], [-33.86383, 151.06277], [-33.86383, 151.06224], [-33.86384, 151.06154], [-33.86385, 151.06112], [-33.86387, 151.06045], [-33.86389, 151.06006], [-33.86394, 151.05961], [-33.86399, 151.05924], [-33.86407, 151.05879], [-33.86414, 151.05843], [-33.86424, 151.058], [-33.86429, 151.05787], [-33.86434, 151.05767], [-33.8644, 151.05741], [-33.86447, 151.05709], [-33.86453, 151.05669], [-33.86454, 151.05658], [-33.86455, 151.05647], [-33.86457, 151.0562], [-33.86458, 151.0559], [-33.86457, 151.05562], [-33.86455, 151.05512], [-33.86452, 151.05483], [-33.86449, 151.05443], [-33.86447, 151.05407], [-33.86446, 151.05343], [-33.86448, 151.05264], [-33.86447, 151.0518], [-33.86446, 151.05137], [-33.86446, 151.05089], [-33.86442, 151.04951], [-33.86442, 151.04948], [-33.86441, 151.04909], [-33.86444, 151.04869], [-33.86443, 151.04818], [-33.86442, 151.04781], [-33.86436, 151.0472], [-33.86434, 151.04705], [-33.86428, 151.04675], [-33.86423, 151.04648], [-33.86417, 151.04619], [-33.86413, 151.04605], [-33.86409, 151.04586], [-33.864, 151.04559], [-33.86392, 151.04536], [-33.86385, 151.04514], [-33.86384, 151.04511], [-33.86374, 151.04488], [-33.86363, 151.04463], [-33.86357, 151.04448], [-33.86356, 151.04447], [-33.86346, 151.04424], [-33.86334, 151.044], [-33.86326, 151.04384], [-33.86314, 151.04364], [-33.86306, 151.04352], [-33.86298, 151.0434], [-33.86287, 151.04324], [-33.86278, 151.04313], [-33.8627, 151.04302], [-33.86258, 151.04286], [-33.86246, 151.0427], [-33.86235, 151.04257], [-33.8622, 151.0424], [-33.86207, 151.04226], [-33.86196, 151.04215], [-33.86182, 151.04202], [-33.86174, 151.04193], [-33.86162, 151.04183], [-33.86153, 151.04175], [-33.86141, 151.04166], [-33.86134, 151.0416], [-33.86124, 151.04153], [-33.86117, 151.04147], [-33.8611, 151.04142], [-33.86099, 151.04134], [-33.85997, 151.04062], [-33.85965, 151.0404], [-33.85904, 151.03998], [-33.85747, 151.03887], [-33.8571, 151.03862], [-33.85611, 151.03793], [-33.85606, 151.03789], [-33.85516, 151.03725], [-33.85371, 151.03624], [-33.85357, 151.03614], [-33.85349, 151.03608], [-33.85322, 151.03589], [-33.85321, 151.03589], [-33.85307, 151.03579], [-33.85276, 151.03558], [-33.85228, 151.03524], [-33.85193, 151.03499], [-33.85183, 151.03492], [-33.85146, 151.03465], [-33.85126, 151.0345], [-33.85123, 151.03448], [-33.85066, 151.03404], [-33.85061, 151.034], [-33.85025, 151.03372], [-33.84995, 151.03345], [-33.84981, 151.03332], [-33.84967, 151.03319], [-33.84953, 151.03306], [-33.84945, 151.03298], [-33.84943, 151.03296], [-33.84928, 151.03281], [-33.84914, 151.03267], [-33.84863, 151.03214], [-33.84858, 151.03208], [-33.84832, 151.03182], [-33.84829, 151.03178], [-33.84804, 151.03153], [-33.84761, 151.03108], [-33.84758, 151.03106], [-33.84726, 151.03073], [-33.84713, 151.0306], [-33.84706, 151.03055], [-33.84694, 151.03046], [-33.8469, 151.03042], [-33.84665, 151.03015], [-33.84646, 151.02995], [-33.84635, 151.02983], [-33.84624, 151.02975], [-33.84614, 151.02966], [-33.84611, 151.02963], [-33.84602, 151.02956], [-33.84597, 151.02951], [-33.8458, 151.02937], [-33.84568, 151.02927], [-33.84563, 151.02922], [-33.8456, 151.02918], [-33.84541, 151.02898], [-33.84531, 151.02887], [-33.84516, 151.02872], [-33.84504, 151.02859], [-33.84487, 151.02842], [-33.8448, 151.02835], [-33.84475, 151.02828], [-33.84472, 151.02824], [-33.8445, 151.02803], [-33.84429, 151.02779], [-33.84402, 151.02751], [-33.84368, 151.02715], [-33.84343, 151.02688], [-33.84315, 151.0266], [-33.84294, 151.0264], [-33.84276, 151.02621], [-33.8426, 151.02606], [-33.84252, 151.02601], [-33.84247, 151.02597], [-33.84232, 151.02582], [-33.84198, 151.02547], [-33.84105, 151.02452], [-33.8404, 151.02382], [-33.83995, 151.02332], [-33.83976, 151.02309], [-33.83952, 151.02277], [-33.83923, 151.02238], [-33.83867, 151.02157], [-33.83827, 151.02088], [-33.83789, 151.02011], [-33.83752, 151.01951], [-33.83737, 151.01926], [-33.83734, 151.01919], [-33.83721, 151.01898], [-33.8372, 151.01896], [-33.83702, 151.01872], [-33.83696, 151.01863], [-33.83693, 151.01851], [-33.83612, 151.01721], [-33.83592, 151.01688], [-33.83584, 151.01677], [-33.83524, 151.01581], [-33.83459, 151.01476], [-33.83452, 151.01464], [-33.83445, 151.01454], [-33.83445, 151.01453], [-33.83404, 151.01395], [-33.8334, 151.0129], [-33.83312, 151.01244], [-33.83309, 151.01239], [-33.8328, 151.01191], [-33.83279, 151.01189], [-33.83276, 151.01185], [-33.83239, 151.01126], [-33.83223, 151.01102], [-33.83173, 151.01019], [-33.83148, 151.00988], [-33.83145, 151.00983], [-33.83143, 151.0098], [-33.8313, 151.00963], [-33.83123, 151.00956], [-33.83107, 151.00938], [-33.83106, 151.00936], [-33.83088, 151.00918], [-33.83069, 151.009], [-33.83045, 151.00881], [-33.8302, 151.00863], [-33.8301, 151.00857], [-33.82994, 151.00848], [-33.82989, 151.00846], [-33.82982, 151.00843], [-33.82914, 151.00807], [-33.82889, 151.00796], [-33.82864, 151.00786], [-33.82825, 151.0077], [-33.82744, 151.0074], [-33.8273, 151.00736], [-33.82719, 151.00733], [-33.82709, 151.0073], [-33.82703, 151.00729], [-33.82695, 151.00729], [-33.82672, 151.00725], [-33.8262, 151.0072], [-33.82559, 151.00717], [-33.82474, 151.00724], [-33.82391, 151.00737], [-33.82293, 151.00767], [-33.82222, 151.00778], [-33.82164, 151.0078], [-33.82154, 151.00779], [-33.82121, 151.00777], [-33.82072, 151.00773], [-33.82018, 151.00756], [-33.81978, 151.00737], [-33.81961, 151.00725], [-33.81946, 151.00715], [-33.81937, 151.00708], [-33.81901, 151.00682], [-33.81861, 151.00649], [-33.81844, 151.00634], [-33.81836, 151.00627], [-33.81829, 151.00621], [-33.81812, 151.00606], [-33.81782, 151.00573], [-33.81767, 151.00554], [-33.81745, 151.00519], [-33.81741, 151.00512]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#ff0000&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;#ff0000&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_14b774d42b52df4d0da4b851913953f2);


        var popup_571698d5eea55c6264b1d3dd2271dcce = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_aa5f8678f604e88b9eb871a15da71cfc = $(`&lt;div id=&quot;html_aa5f8678f604e88b9eb871a15da71cfc&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;TRANSIT (34 mins)&lt;/div&gt;`)[0];
                popup_571698d5eea55c6264b1d3dd2271dcce.setContent(html_aa5f8678f604e88b9eb871a15da71cfc);



        poly_line_bf10cd2779b943249f29cd6067d1bc50.bindPopup(popup_571698d5eea55c6264b1d3dd2271dcce)
        ;




            var poly_line_2b09531cfc29984f813f16186fef9deb = L.polyline(
                [[-33.81751, 151.00504], [-33.81702, 151.00489], [-33.81699, 151.00491], [-33.81691, 151.00469], [-33.81688, 151.00458], [-33.81686, 151.0045], [-33.81685, 151.00445], [-33.81681, 151.00432], [-33.81677, 151.00418], [-33.81675, 151.00411], [-33.81674, 151.00408], [-33.81674, 151.00407], [-33.81672, 151.004], [-33.81668, 151.00384], [-33.81666, 151.00374], [-33.81663, 151.00364], [-33.81662, 151.00357], [-33.81661, 151.00353], [-33.8166, 151.00347], [-33.81654, 151.00343], [-33.81647, 151.00339], [-33.81637, 151.00334], [-33.81633, 151.00332], [-33.8163, 151.0033], [-33.81626, 151.00328], [-33.81619, 151.00325], [-33.81594, 151.00319], [-33.81555, 151.0031], [-33.81552, 151.00309], [-33.81549, 151.00307], [-33.81543, 151.00302], [-33.81536, 151.00298], [-33.81531, 151.00294], [-33.81523, 151.00292], [-33.81519, 151.00291], [-33.81516, 151.00291], [-33.81513, 151.00291], [-33.81508, 151.00292], [-33.81506, 151.00284], [-33.81505, 151.00279], [-33.815, 151.00258], [-33.81495, 151.00233]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#0000ff&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;#0000ff&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_14b774d42b52df4d0da4b851913953f2);


        var popup_4b0f1b941dcadbe1f13964402278bdf9 = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_590a5f25a8a9f43c7b491784e7c41cfc = $(`&lt;div id=&quot;html_590a5f25a8a9f43c7b491784e7c41cfc&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;WALKING (7 mins)&lt;/div&gt;`)[0];
                popup_4b0f1b941dcadbe1f13964402278bdf9.setContent(html_590a5f25a8a9f43c7b491784e7c41cfc);



        poly_line_2b09531cfc29984f813f16186fef9deb.bindPopup(popup_4b0f1b941dcadbe1f13964402278bdf9)
        ;



&lt;/script&gt;
&lt;/html&gt;" width="800" height="600"style="border:none !important;" "allowfullscreen" "webkitallowfullscreen" "mozallowfullscreen"></iframe>


