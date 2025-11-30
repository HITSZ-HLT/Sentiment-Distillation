while getopts ':c:b:v:z:' opt
do
    case $opt in
        c) CUDA_IDS="$OPTARG" ;;
        b) subname="$OPTARG" ;;
        v) model_version="$OPTARG" ;;
        z) z_value="$OPTARG" ;;
        ?)  
        exit 1;;
    esac
done

if [ -z "${z_value}" ]; then
    z_value=4
fi

tasks=("sc/imdb" "sc/yelp2" "sc/sst2" "sc/twitter" "multifaced/irony18" "multifaced/tweeteval" "multifaced/pstance" "multifaced/intimacy" )


# 循环执行指令
for task in "${tasks[@]}"; do
    if [[ "$task" == "sc/imdb" || "$task" == "sc/yelp2" ]]; then
        z_value=1
    else
        z_value=4
    fi
    bash/multifaced.sh -c ${CUDA_IDS} -d $task -b ${subname} -z ${z_value} -v ${model_version} -s 42
done
