import requests

'''
url = 'https://project-glovo-graph-simulation.onrender.com/'
res = requests.get(url=url, timeout=10, params={
    'dbraw': 'project-glovo',
    'dbrendered': 'rendered',
    'webhook': 'https://project-glovo-backend-background.onrender.com/webhook/fedcd325-8364-4b59-9f8a-ad335374938f/'
})

print(res.url)
'''

url = 'http://localhost:10000/cycle/'
r = requests.get(url, headers={"API_SECRET_KEY": "secret_key_0026"})
print(r.content)