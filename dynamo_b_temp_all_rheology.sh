#!/usr/bin/env bash

# initializes tunable parameters.
forced_eqna_patterns=(*i_lageos1_lageos2_sta*tides *i_lageos1_lageos2_ste*tides *i_lageos1_starlette_ste*tides *i_lageos2_starlette_ste*tides *y_lageos1_lageos2_starlette_ste*tides)
bounds=(low high)

# Initializes directories.
root=/work/GRGS/users/rousselm/dynamo/
rheology_directory=${root}/rheology
dird_model=${root}/models/DIRD_rheology
G_constraints_directory=${root}/G_constraints
solution_directory=${rheology_directory}/solution_fix_asthenosphere
listing_directory=${rheology_directory}/listing_fix_asthenosphere
mkdir -p "$solution_directory"
mkdir -p "$listing_directory"

for forced_eqna_pattern in "${forced_eqna_patterns[@]}"; do

	for lam_bound in "${bounds[@]}"; do

		for ldm_bound in "${bounds[@]}"; do

			# Second solves every normal equation for current framework without constraint.
			find ${rheology_directory}/eqna/${forced_eqna_pattern} -type f | while read -r file; do
				
				file_name=$(basename "$file")
				
				# Prevents from overwritting.
				if [[ ! -f "${solution_directory}/${file_name}" ]]; then

				# Accumulates per satellite and reduces the useless unknowns.
				bash exe_dynamo_b_local.sh \
					-dir /work/GRGS/users/rousselm/dynamo/models/DIRB_rheology_${lam_bound}_lam_${ldm_bound}_ldm \
					-b "${file}" \
					-out "/work/GRGS/users/rousselm/dynamo/rheology/listing_fix_asthenosphere/dyb_out_${file_name}_${lam_bound}_lam_${ldm_bound}_ldm" \
					-r "/work/GRGS/users/rousselm/dynamo/rheology/eqna_fix_asthenosphere/${file_name}_${lam_bound}_lam_${ldm_bound}_ldm" \
					-lT 1400

						
				fi
					

			done

		done
		
	done

done
