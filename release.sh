#!/usr/bin/env bash
# Usage: ./release.sh <version>  e.g. ./release.sh 0.1.9
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <version>"
  echo "Example: $0 0.1.9"
  exit 1
fi

VERSION=$1
TAG="v$VERSION"

# Bump version in pyproject.toml
sed -i "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml

# Commit, tag, and push
git add pyproject.toml
git commit -m "Bump version to $VERSION"
git tag -a "$TAG" -m "Release $TAG"
git push origin main
git push origin "$TAG"

echo "Released $TAG — PyPI publish workflow will start automatically."
echo "Monitor: https://github.com/Aswincloud/ttperf/actions"
echo "PyPI: https://pypi.org/project/ttperf/$VERSION/"
