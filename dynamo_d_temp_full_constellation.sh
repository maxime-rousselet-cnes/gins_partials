#!/usr/bin/env bash

# initializes tunable parameters.
G_constraints=(constraints_G_trend_and_acceleration_and_annual constraints_G_trend_and_acceleration constraints_G_trend_and_annual constraints_G_trend)
sigma_exponents=("12" "13" "14")
forced_eqna_pattern=*i_lageos1_lageos2_sta*ste*

# Initializes directories.
root=/work/GRGS/users/rousselm/dynamo/
rheology_directory=${root}/rheology
dird_model=${root}/models/DIRD_rheology_fix_asthenosphere
G_constraints_directory=${root}/G_constraints
solution_directory=${rheology_directory}/solution_full_constellation
listing_directory=${rheology_directory}/listing_full_constellation
mkdir -p "$solution_directory"
mkdir -p "$listing_directory"

# Initializes constraint files depending on the sigma exponent.
for exponent in "${sigma_exponents[@]}"; do

    constraint_exponent_subdirectory=${G_constraints_directory}/sigma_E-${exponent}
    mkdir -p "$constraint_exponent_subdirectory"
    find "$G_constraints_directory" -maxdepth 1 -type f -exec cp {} "$constraint_exponent_subdirectory"/ \;
    cd "$constraint_exponent_subdirectory"
    sed -i "s/E-20/E-${exponent}/g" *
    cd ..

done



# Second solves every normal equation for current framework without constraint.
find ${rheology_directory}/eqna/${forced_eqna_pattern} -type f | while read -r file; do
	
	file_name=$(basename "$file")
	
	# Prevents from overwritting.
	if [[ ! -f "${solution_directory}/${file_name}" ]]; then
	
		exe_dynamo_d \
			-dir "$dird_model" -b "$file" \
			-out "${listing_directory}/dyd_out_${file_name}" \
			-s "${solution_directory}/${file_name}"
			
	fi
		

done

cd "$root/.."

# Third loops on constraints.
for exponent in "${sigma_exponents[@]}"; do

	solution_exponent_directory=${solution_directory}/sigma_E-${exponent}
	listing_exponent_directory=${listing_directory}/sigma_E-${exponent}
	mkdir -p "$solution_exponent_directory"
	mkdir -p "$listing_exponent_directory"

	for G_constraint in "${G_constraints[@]}"; do

		solution_G_subdirectory=${solution_exponent_directory}/${G_constraint}
		listing_G_subdirectory=${listing_exponent_directory}/${G_constraint}
		mkdir -p "$solution_G_subdirectory"
		mkdir -p "$listing_G_subdirectory"
		
		bash submit_dynamo_loop.sh \
			"$rheology_directory" \
			"$forced_eqna_pattern" \
			"$listing_G_subdirectory" \
			"$dird_model" \
			"$G_constraints_directory" \
			"$exponent" \
			"$G_constraint" \
			"$solution_G_subdirectory"
		
		sleep 10

	done

done