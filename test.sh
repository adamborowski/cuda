sudo Debug/CUDAProj | egrep -o '[0-9]+ ->' | cut -d' ' -f1 | sort | uniq -u | wc -l
