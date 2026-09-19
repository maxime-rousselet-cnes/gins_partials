#!/usr/bin/env bash

# initializes tunable parameters.
forced_eqna_pattern=*

# Initializes directories.
root=/work/GRGS/users/rousselm/dynamo/
rheolody_directory=${root}/rheology
G_constraints_directory=${root}/G_constraints
solution_directory=${rheolody_directory}/solution_all_constellations
listing_directory=${rheolody_directory}/listing_all_constellations
mkdir -p "$solution_directory"
mkdir -p "$listing_directory"

# Initializes constraint files depending on the sigma exponent.
constraint_exponent_subdirectory=${G_constraints_directory}/sigma_E-13
mkdir -p "$constraint_exponent_subdirectory"
cp "${G_constraints_directory}/constraints_G_trend_and_acceleration_and_annual" "${constraint_exponent_subdirectory}/."
cd "$constraint_exponent_subdirectory"
sed -i "s/E-20/E-13/g" constraints_G_trend_and_acceleration_and_annual
cd "$root/.."

bash submit_dynamo_loop.sh \
    "$rheology_directory" \
    "*" \
    "$listing_directory" \
    "${root}/models/DIRD_rheology" \
    "$G_constraints_directory" \
    "13" \
    "constraints_G_trend_and_acceleration_and_annual" \
    "$solution_directory"
