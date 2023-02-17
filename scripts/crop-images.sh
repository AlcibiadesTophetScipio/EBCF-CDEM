########### astgtmv003 145 x16
## crop
# resize_shape="561x577+210+184"
# regexr="/data/syao/Exps/vedo-show/astgtmv003*x16_145.png"
# suffix="crop.png"
## amplify-1
# resize_shape="121x141+488+188"
# regexr="/data/syao/Exps/vedo-show/astgtmv003*x16_145.png"
# suffix="amp1.png"
## amplify-2
# resize_shape="101x94+326+407"
# regexr="/data/syao/Exps/vedo-show/astgtmv003*x16_145.png"
# suffix="amp2.png"

########### astgtmv003 126 x8
## crop
# resize_shape="584x543+157+183"
# regexr="/data/syao/Exps/vedo-show/*/astgtmv003*x8_126.png"
# suffix="crop.png"
## amplify-1
# resize_shape="101x115+324+400"
# regexr="/data/syao/Exps/vedo-show/*/astgtmv003*x8_126.png"
# suffix="amp1.png"

########### pyrenees 22 x16
## crop
# resize_shape="594x581+196+209"
# regexr="/data/syao/Exps/vedo-show/pyrenees_x16_v10.0/pyrenees*x16_22.png"
# suffix="crop.png"
## amplify-1
# resize_shape="109x96+421+443"
# regexr="/data/syao/Exps/vedo-show/pyrenees_x16_v10.0/pyrenees*x16_22.png"
# suffix="amp1.png"
## amplify-2
# resize_shape="58x77+595+646"
# regexr="/data/syao/Exps/vedo-show/pyrenees_x16_v10.0/pyrenees*x16_22.png"
# suffix="amp2.png"

########### pyrenees 22 x8
## crop
# resize_shape="600x583+193+208"
# regexr="/data/syao/Exps/vedo-show/pyrenees_x8_v5.0/pyrenees*x8_22.png"
# suffix="crop.png"
## amplify-1
# resize_shape="60x63+480+534"
# regexr="/data/syao/Exps/vedo-show/pyrenees_x8_v5.0/pyrenees*x8_22.png"
# suffix="amp1.png"
## amplify-2
# resize_shape="82x94+461+257"
# regexr="/data/syao/Exps/vedo-show/pyrenees_x8_v5.0/pyrenees*x8_22.png"
# suffix="amp2.png"

########### pyrenees 22 x4
## crop
# resize_shape="599x587+195+207"
# regexr="/data/syao/Exps/vedo-show/pyrenees_x4_v2.0/pyrenees*x4_22.png"
# suffix="crop.png"
## amplify-1
# resize_shape="54x81+633+534"
# regexr="/data/syao/Exps/vedo-show/pyrenees_x4_v2.0/pyrenees*x4_22.png"
# suffix="amp1.png"
## amplify-2
# resize_shape="43x44+292+322"
# regexr="/data/syao/Exps/vedo-show/pyrenees_x4_v2.0/pyrenees*x4_22.png"
# suffix="amp2.png"

########### asgtmv003 22 x4
## crop
# resize_shape="584x544+157+183"
# regexr="/data/syao/Exps/vedo-show/astgtmv003_x4_v20.0/astgtmv003*x4_126.png"
# suffix="crop.png"
## amplify-1
# resize_shape="60x50+596+475"
# regexr="/data/syao/Exps/vedo-show/astgtmv003_x4_v20.0/astgtmv003*x4_126.png"
# suffix="amp1.png"
## amplify-2
# resize_shape="62x50+563+257"
# regexr="/data/syao/Exps/vedo-show/astgtmv003_x4_v20.0/astgtmv003*x4_126.png"
# suffix="amp2.png"

########### asgtmv003 22 x8
## crop
# resize_shape="585x542+157+182"
# regexr="/data/syao/Exps/vedo-show/astgtmv003_x8_v40.0/astgtmv003*x8_126.png"
# suffix="crop.png"
## amplify-1
# resize_shape="164x153+435+307"
# regexr="/data/syao/Exps/vedo-show/astgtmv003_x8_v40.0/astgtmv003*x8_126.png"
# suffix="amp1.png"
## amplify-2
# resize_shape="126x114+188+346"
# regexr="/data/syao/Exps/vedo-show/astgtmv003_x8_v40.0/astgtmv003*x8_126.png"
# suffix="amp2.png"

########### asgtmv003 22 x16
## crop
resize_shape="581x545+162+180"
regexr="/data/syao/Exps/vedo-show/astgtmv003_x16_v80.0/astgtmv003*x16_126.png"
suffix="crop.png"
## amplify-1
# resize_shape="112x93+427+406"
# regexr="/data/syao/Exps/vedo-show/astgtmv003_x16_v80.0/astgtmv003*x16_126.png"
# suffix="amp1.png"
## amplify-2
# resize_shape="196x101+295+218"
# regexr="/data/syao/Exps/vedo-show/astgtmv003_x16_v80.0/astgtmv003*x16_126.png"
# suffix="amp2.png"


for image_file in $regexr;
do
    new_file=${image_file%.png}-${suffix}
    # convert ${source_img} -crop ${height}x${width}+${left_top_x}+${left_top_y} ${target_img}
    crop_cmd="convert ${image_file} -crop ${resize_shape} ${new_file}"
    echo ${crop_cmd}
    eval ${crop_cmd}
done

