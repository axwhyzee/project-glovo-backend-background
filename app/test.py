import requests

url = 'https://project-glovo-graph-simulation.onrender.com/'
res = requests.get(url=url, timeout=10, params={
    'dbraw': 'project-glovo',
    'dbrendered': 'rendered',
    'webhook': 'https://project-glovo-backend-background.onrender.com/webhook/fedcd325-8364-4b59-9f8a-ad335374938f/'
})

print(res.url)
