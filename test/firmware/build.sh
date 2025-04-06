#!/bin/bash
for name in simple print spigot; do
    for c in true false; do
        $1 build -Dname=$name -Dcompressed=$c
    done
done
