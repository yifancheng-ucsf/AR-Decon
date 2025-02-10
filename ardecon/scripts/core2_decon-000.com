#!/bin/sh

pos_args=()
while [ $OPTIND -le "$#" ]
do
    if getopts o:n: flag
    then
        case "${flag}" in
            o) output=${OPTARG};;
            n) iter=${OPTARG};;
        esac
    else
        pos_args+=("${!OPTIND}")
        ((OPTIND++))
    fi
done

input=${pos_args[0]}
if [ ! -z "${output}" ]
then
    root=$output
else
    root=$(basename $input .mrc)
fi
tf=${pos_args[1]}

if [ -z "${iter}" ]
then
    iter=50
fi

echo $input
echo $root
echo $tf

echo Running on `hostname` >"${root}_Decon.log"
#Setting run time environment...
#command file for core2_decon
( time core2_decon \
 "${input}" \
 "${root}_Decon.mrc" \
 "${tf}" \
 -alpha=10000 -lamratio=0:1 -lamf=0.5 -lampc=0 -lampos=1 \
 -lamsmooth=100 -laml2=0 -cuth=0.001 -na=1.4 -nimm=1.512 -ncycl=${iter} \
 -nzpad=512 -omega=0.8 -sub=1:1:1:1:1 -tol=0.0001 -np=4 \
 -guess="" \
 -linesearch="2014" -regtype="ma" ) \
 >>"${root}_Decon.log" 2>&1 ||
 ( time core2_decon_static \
 "${input}" \
 "${root}_Decon.mrc" \
 "${tf}" \
 -alpha=10000 -lamratio=0:1 -lamf=0.5 -lampc=0 -lampos=1 \
 -lamsmooth=100 -laml2=0 -cuth=0.001 -na=1.4 -nimm=1.512 -ncycl=${iter} \
 -nzpad=512 -omega=0.8 -sub=1:1:1:1:1 -tol=0.0001 -np=4 \
 -guess="" \
 -linesearch="2014" -regtype="ma" ) \
 >>"${root}_Decon.log" 2>&1
