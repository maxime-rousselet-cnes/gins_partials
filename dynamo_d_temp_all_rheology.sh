#!/usr/bin/env bash

# initializes tunable parameters.
G_constraints=(constraints_G_trend_and_acceleration_and_annual constraints_G_trend_and_acceleration)
sigma_exponents=("13")

root=/work/GRGS/users/rousselm/dynamo/
rheology_directory=${root}/rheology
dird_model=${root}/models/DIRD_rheology_fix_asthenosphere
G_constraints_directory=${root}/G_constraints
solution_directory=${rheology_directory}/solution_all_rheology
listing_directory=${rheology_directory}/listing_all_rheology
mkdir -p "$solution_directory"
mkdir -p "$listing_directory"

for G_constraint in "${G_constraints[@]}"; do

	solution_G_subdirectory=${solution_directory}/${G_constraint}
	listing_G_subdirectory=${listing_directory}/${G_constraint}
	mkdir -p "$solution_G_subdirectory"
	mkdir -p "$listing_G_subdirectory"
	
	bash submit_dynamo_loop.sh \
		"$rheology_directory" \
		"../eqna_fix_asthenosphere/*" \
		"$listing_G_subdirectory" \
		"$dird_model" \
		"$G_constraints_directory" \
		"13" \
		"$G_constraint" \
		"$solution_G_subdirectory"

done
