#!/usr/bin/env bash

# initializes tunable parameters.
G_constraints=(constraints_G_trend_and_acceleration_and_annual constraints_G_trend_and_annual constraints_G_trend_and_acceleration constraints_G_trend)
sigma_exponents=("16" "19")
forced_eqna_pattern=*ajisai_lageos1_st*

# Initializes directories.
root=/work/GRGS/users/rousselm/dynamo/
rheolody_directory=${root}/rheology
dird_model=${root}/models/DIRD_rheology
G_constraints_directory=${root}/G_constraints
solution_directory=${rheolody_directory}/solution
listing_directory=${rheolody_directory}/listing
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

# Iterates on frameworks (2^4 = 16 cases).
for fix_alpha in true false; do

    solution_alpha_directory=${solution_directory}/fix_alpha_${fix_alpha}
    listing_alpha_directory=${listing_directory}/fix_alpha_${fix_alpha}
    mkdir -p "$solution_alpha_directory"
    mkdir -p "$listing_alpha_directory"

	for fix_log10_delta in true false; do

        solution_log10_delta_directory=${solution_alpha_directory}/fix_log10_delta_${fix_log10_delta}
        listing_log10_delta_directory=${listing_alpha_directory}/fix_log10_delta_${fix_log10_delta}
        mkdir -p "$solution_log10_delta_directory"
        mkdir -p "$listing_log10_delta_directory"
			
		for fix_log10_tau_m in true false; do

            solution_log10_tau_m_directory=${solution_log10_delta_directory}/fix_log10_tau_m_${fix_log10_tau_m}
            listing_log10_tau_m_directory=${listing_log10_delta_directory}/fix_log10_tau_m_${fix_log10_tau_m}
            mkdir -p "$solution_log10_tau_m_directory"
            mkdir -p "$listing_log10_tau_m_directory"
		
			for fix_G in true false; do

                solution_G_directory=${solution_log10_tau_m_directory}/fix_G_${fix_G}
                listing_G_directory=${listing_log10_tau_m_directory}/fix_G_${fix_G}
                mkdir -p "$solution_G_directory"
                mkdir -p "$listing_G_directory"

                # First creates a DIRD adapted to the current framework.
                DIRD_variant=${listing_G_directory}/DIRD_rheology_variant
				cp "$dird_model" "$DIRD_variant"
					
				if [[ "$fix_G" == true ]]; then
				
					sed -i '/FIN    LECBDIR/i\
FIX [G???????????????????????]
' "$DIRD_variant"

				fi		
					
				if [[ "$fix_alpha" == true ]]; then
				
					sed -i '/FIN    LECBDIR/i\
FIX [LAM?????????????????????]
' "$DIRD_variant"
					
				fi			
					
				if [[ "$fix_log10_delta" == true ]]; then
				
					sed -i '/FIN    LECBDIR/i\
FIX [LDM?????????????????????]
' "$DIRD_variant"
					
				fi			
					
				if [[ "$fix_log10_tau_m" == true ]]; then
				
					sed -i '/FIN    LECBDIR/i\
FIX [LTM?????????????????????]
' "$DIRD_variant"
					
				fi

                # Second solves every normal equation for current framework without constraint.
                find ${rheolody_directory}/eqna/${forced_eqna_pattern} -type f | while read -r file; do
                    
	                file_name=$(basename "$file")
	                
	                # Prevents from overwritting.
	                if [[ ! -f "${solution_G_directory}/${file_name}" ]]; then
	                
						exe_dynamo_d \
							-dir "$DIRD_variant" -b "$file" \
							-out "${listing_G_directory}/dyd_out_${file_name}" \
							-s "${solution_G_directory}/${file_name}"
							
					fi
						
                
                done

                
                if [[ "$fix_G" == false ]]; then

                    # Third loops on constraints.
                    for exponent in "${sigma_exponents[@]}"; do

                        solution_exponent_directory=${solution_G_directory}/sigma_E-${exponent}
                        listing_exponent_directory=${listing_G_directory}/sigma_E-${exponent}
                        mkdir -p "$solution_exponent_directory"
                        mkdir -p "$listing_exponent_directory"

                        for G_constraint in "${G_constraints[@]}"; do

                            solution_G_subdirectory=${solution_exponent_directory}/${G_constraint}
                            listing_G_subdirectory=${listing_exponent_directory}/${G_constraint}
                            mkdir -p "$solution_G_subdirectory"
                            mkdir -p "$listing_G_subdirectory"

                            # Solves every normal equation for current framework and constraint.
                            find ${rheolody_directory}/eqna/${forced_eqna_pattern} -type f | while read -r file; do

                                file_name=$(basename "$file")
                                
								# Prevents from overwritting.
								if [[ ! -f "${listing_G_subdirectory}/${file_name}" ]]; then
	                
									exe_dynamo_d \
										-dir "$DIRD_variant" -b "$file" \
										-cont "${G_constraints_directory}/sigma_E-${exponent}/${G_constraint}" \
										-eqna_cont "${listing_G_subdirectory}/eqna_cont" \
										-out "${listing_G_subdirectory}/dyd_out_${file_name}" \
										-s "${solution_G_subdirectory}/${file_name}"
								
								fi

                            done

                        done

                    done

                fi

			done
			
		done
			
	done
		
done

cd "$root"
