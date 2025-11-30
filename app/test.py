import base64

spring_secret = "U29tZV9zdXBlcl9zZWN1cmVfYW5kX2xvbmdfYmFzZTY0X2VuY29kZWRfc2VjcmV0X2tleV9mb3JfSlNXVDEyMw=="
decoded = base64.b64decode(spring_secret)
print(decoded)
print(len(decoded))
