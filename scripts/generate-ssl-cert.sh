#!/bin/bash
#
# Generate Self-Signed SSL Certificate for Development
#
# Usage:
#   ./scripts/generate-ssl-cert.sh
#
# For production, replace with proper certificates from Let's Encrypt or CA
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Generating self-signed SSL certificate for development...${NC}"

# Create ssl directory
SSL_DIR="./ssl"
mkdir -p "$SSL_DIR"

# Certificate details
DAYS=365
COUNTRY="US"
STATE="California"
CITY="San Francisco"
ORG="Agentic Backend"
OU="Development"
CN="localhost"

# Generate private key
echo -e "${YELLOW}Generating private key...${NC}"
openssl genrsa -out "$SSL_DIR/key.pem" 2048

# Generate certificate signing request
echo -e "${YELLOW}Generating certificate signing request...${NC}"
openssl req -new -key "$SSL_DIR/key.pem" -out "$SSL_DIR/csr.pem" \
  -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/OU=$OU/CN=$CN"

# Generate self-signed certificate
echo -e "${YELLOW}Generating self-signed certificate...${NC}"
openssl x509 -req -days $DAYS \
  -in "$SSL_DIR/csr.pem" \
  -signkey "$SSL_DIR/key.pem" \
  -out "$SSL_DIR/cert.pem" \
  -extfile <(printf "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1")

# Clean up CSR
rm "$SSL_DIR/csr.pem"

# Set permissions
chmod 600 "$SSL_DIR/key.pem"
chmod 644 "$SSL_DIR/cert.pem"

echo -e "${GREEN}✓ SSL certificate generated successfully!${NC}"
echo ""
echo "Certificate location: $SSL_DIR/cert.pem"
echo "Private key location: $SSL_DIR/key.pem"
echo "Valid for: $DAYS days"
echo ""
echo -e "${YELLOW}⚠️  This is a self-signed certificate for DEVELOPMENT only!${NC}"
echo -e "${YELLOW}⚠️  For production, use Let's Encrypt or a proper CA certificate${NC}"
echo ""
echo "Certificate details:"
openssl x509 -in "$SSL_DIR/cert.pem" -text -noout | grep -E "Subject:|Issuer:|Not Before|Not After|DNS:"
