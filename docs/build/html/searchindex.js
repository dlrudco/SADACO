Search.setIndex({docnames:["SADACO/Datasets","SADACO/index","SADACO_WEB/codes","SADACO_WEB/index","SADACO_WEB/markdown","apis/sadaco.apis","apis/sadaco.apis.contrastive","apis/sadaco.apis.explain","apis/sadaco.apis.logger","apis/sadaco.apis.losses","apis/sadaco.apis.models","apis/sadaco.apis.traintest","index","modules","sadaco","sadaco.dataman","sadaco.pipelines","sadaco.utils"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["SADACO/Datasets.md","SADACO/index.rst","SADACO_WEB/codes.md","SADACO_WEB/index.rst","SADACO_WEB/markdown.md","apis/sadaco.apis.rst","apis/sadaco.apis.contrastive.rst","apis/sadaco.apis.explain.rst","apis/sadaco.apis.logger.rst","apis/sadaco.apis.losses.rst","apis/sadaco.apis.models.rst","apis/sadaco.apis.traintest.rst","index.rst","modules.rst","sadaco.rst","sadaco.dataman.rst","sadaco.pipelines.rst","sadaco.utils.rst"],objects:{"":{sadaco:[14,0,0,"-"]},"sadaco.apis":{contrastive:[6,0,0,"-"],explain:[7,0,0,"-"],logger:[8,0,0,"-"],losses:[9,0,0,"-"],models:[10,0,0,"-"],traintest:[11,0,0,"-"]},"sadaco.apis.contrastive":{train_byol:[6,0,0,"-"],train_supcon:[6,0,0,"-"],trainer_contrastive:[6,0,0,"-"]},"sadaco.apis.contrastive.train_supcon":{move_device:[6,1,1,""],train_mixcon_epoch:[6,1,1,""]},"sadaco.apis.contrastive.trainer_contrastive":{ContrastTrainer:[6,2,1,""]},"sadaco.apis.contrastive.trainer_contrastive.ContrastTrainer":{attach_extractor:[6,3,1,""],wrap_model:[6,3,1,""]},"sadaco.apis.explain":{explainer:[7,0,0,"-"],hookman:[7,0,0,"-"],visualize:[7,0,0,"-"]},"sadaco.apis.explain.explainer":{BaseExplainer:[7,2,1,""],GradcamExplainer:[7,2,1,""]},"sadaco.apis.explain.explainer.GradcamExplainer":{forward:[7,3,1,""]},"sadaco.apis.explain.hookman":{FGHandler:[7,2,1,""],get_last_conv_name:[7,1,1,""]},"sadaco.apis.explain.hookman.FGHandler":{eval:[7,3,1,""],forward:[7,3,1,""],get_all_features:[7,3,1,""],get_all_grads:[7,3,1,""],get_features:[7,3,1,""],get_grads:[7,3,1,""],remove_handlers:[7,3,1,""],reset_all:[7,3,1,""],to:[7,3,1,""],train:[7,3,1,""]},"sadaco.apis.explain.visualize":{figure_to_array:[7,1,1,""],get_input_img:[7,1,1,""],load_input:[7,1,1,""],min_max_scale:[7,1,1,""],spec_display:[7,1,1,""]},"sadaco.apis.logger":{basic:[8,0,0,"-"]},"sadaco.apis.logger.basic":{BaseLogger:[8,2,1,""]},"sadaco.apis.logger.basic.BaseLogger":{generate_expid:[8,3,1,""],log:[8,3,1,""]},"sadaco.apis.losses":{BasicLoss:[9,0,0,"-"],ContrastiveLoss:[9,0,0,"-"],CustomLoss:[9,0,0,"-"]},"sadaco.apis.losses.CustomLoss":{mixup_criterion:[9,1,1,""]},"sadaco.apis.models":{build_model:[10,0,0,"-"],cbam:[10,0,0,"-"],cnn_moe:[10,0,0,"-"],compile_trt:[10,0,0,"-"],custom:[10,0,0,"-"],resnet:[10,0,0,"-"]},"sadaco.apis.models.build_model":{build_model:[10,1,1,""]},"sadaco.apis.models.cbam":{logsumexp_2d:[10,1,1,""]},"sadaco.apis.models.cnn_moe":{cnn_moe:[10,1,1,""]},"sadaco.apis.models.compile_trt":{compile:[10,1,1,""]},"sadaco.apis.models.custom":{custom_model:[10,1,1,""]},"sadaco.apis.models.resnet":{ResidualNet:[10,1,1,""],conv3x3:[10,1,1,""]},"sadaco.apis.traintest":{common:[11,0,0,"-"],demo:[11,0,0,"-"],eval:[11,0,0,"-"],preprocessings:[11,0,0,"-"],train:[11,0,0,"-"],trainer_base:[11,0,0,"-"]},"sadaco.apis.traintest.common":{load_input:[11,1,1,""]},"sadaco.apis.traintest.demo":{demo_helper:[11,2,1,""]},"sadaco.apis.traintest.demo.demo_helper":{do_explanation:[11,3,1,""],do_inference:[11,3,1,""]},"sadaco.apis.traintest.eval":{move_device:[11,1,1,""],test_basic_epoch:[11,1,1,""]},"sadaco.apis.traintest.preprocessings":{Preprocessor:[11,2,1,""],normalize:[11,2,1,""],stft2meldb:[11,2,1,""]},"sadaco.apis.traintest.preprocessings.Preprocessor":{add_module:[11,3,1,""],to:[11,3,1,""]},"sadaco.apis.traintest.preprocessings.normalize":{to:[11,3,1,""]},"sadaco.apis.traintest.preprocessings.stft2meldb":{to:[11,3,1,""]},"sadaco.apis.traintest.train":{move_device:[11,1,1,""],train_basic_epoch:[11,1,1,""]},"sadaco.apis.traintest.trainer_base":{BaseTrainer:[11,2,1,""],build_criterion:[11,1,1,""],build_dataloader:[11,1,1,""],build_optimizer:[11,1,1,""]},"sadaco.apis.traintest.trainer_base.BaseTrainer":{build_dataloader:[11,3,1,""],build_dataset:[11,3,1,""],build_logger:[11,3,1,""],build_model:[11,3,1,""],build_optimizer:[11,3,1,""],parallel:[11,3,1,""],prepare_kfold:[11,3,1,""],reset_trainer:[11,3,1,""],resume:[11,3,1,""],test:[11,3,1,""],train:[11,3,1,""],train_epoch:[11,3,1,""],train_kfold:[11,3,1,""],validate:[11,3,1,""],validate_epoch:[11,3,1,""]},"sadaco.dataman":{base:[15,0,0,"-"],build_dataset:[15,0,0,"-"],loader:[15,0,0,"-"],sampler:[15,0,0,"-"]},"sadaco.pipelines":{ICBHI:[16,0,0,"-"],build_modules:[16,0,0,"-"],scheduler:[16,0,0,"-"]},"sadaco.pipelines.ICBHI":{ICBHI_Basic_Trainer:[16,2,1,""],ICBHI_Contrast_Trainer:[16,2,1,""],main:[16,1,1,""],parse_configs:[16,1,1,""]},"sadaco.pipelines.ICBHI.ICBHI_Basic_Trainer":{build_dataset:[16,3,1,""],train_epoch:[16,3,1,""],validate_epoch:[16,3,1,""]},"sadaco.pipelines.ICBHI.ICBHI_Contrast_Trainer":{build_dataset:[16,3,1,""],train_epoch:[16,3,1,""],validate_epoch:[16,3,1,""]},"sadaco.pipelines.build_modules":{build_criterion:[16,1,1,""],build_dataloader:[16,1,1,""],build_optimizer:[16,1,1,""]},"sadaco.pipelines.scheduler":{BaseScheduler:[16,2,1,""]},"sadaco.pipelines.scheduler.BaseScheduler":{step:[16,3,1,""]},"sadaco.utils":{config_parser:[17,0,0,"-"],misc:[17,0,0,"-"],stats:[17,0,0,"-"],web_utils:[17,0,0,"-"]},"sadaco.utils.config_parser":{ArgsParser:[17,2,1,""],join:[17,1,1,""],parse_config_dict:[17,1,1,""],parse_config_obj:[17,1,1,""]},"sadaco.utils.config_parser.ArgsParser":{add_argument:[17,3,1,""],get_args:[17,3,1,""]},"sadaco.utils.misc":{min_max_scale:[17,1,1,""],seed_everything:[17,1,1,""]},"sadaco.utils.stats":{ICBHI_Metrics:[17,2,1,""],print_stats:[17,1,1,""]},"sadaco.utils.stats.ICBHI_Metrics":{__init__:[17,3,1,""],get_mixup_stats:[17,3,1,""],get_precision_recall_fbeta:[17,3,1,""],get_stats:[17,3,1,""],reset_metrics:[17,3,1,""],update_lists:[17,3,1,""],update_mixup_stats:[17,3,1,""]},"sadaco.utils.web_utils":{get_configs:[17,1,1,""],load_config:[17,1,1,""]},sadaco:{apis:[5,0,0,"-"],dataman:[15,0,0,"-"],pipelines:[16,0,0,"-"],utils:[17,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method"},terms:{"100":0,"1000hz":0,"100hz":0,"112":0,"1120":7,"126":0,"128":[10,11],"139802383320832":[],"139802383470704":[],"139802383470752":[],"139802383471280":[],"139802383471328":[],"139802399886352":[],"139802399886496":[],"139802400349632":[],"139931667958800":10,"139931668285856":[6,11],"139931668285904":[6,10,11],"139931668285952":[6,11],"139931668286000":[6,11],"139931683126288":[11,17],"139931683126432":17,"139931683585472":[6,11],"140366858839376":[],"140529033958832":16,"16000":[7,11],"1864":0,"1952":4,"1982":4,"1984":4,"1986":4,"200hz":0,"2017":1,"224":10,"256":10,"336":0,"3x3":10,"500hz":0,"506":0,"512":10,"6898":0,"886":0,"920":0,"bubl\u00e9":4,"char":8,"class":[6,7,8,10,11,16,17],"default":[10,11,17],"final":4,"float":[10,17],"int":[6,10,11,17],"long":4,"new":4,"return":[11,17],"short":[0,4],"true":[4,6,7,11,17],"var":4,And:4,But:0,For:0,One:17,Ones:4,The:[0,4,17],There:[0,4],These:0,With:4,__init__:[11,17],_description_:11,_summary_:11,_type_:11,abcdefghijklmnopqrstuvwxyz0123456789:8,abdullah:0,abil:17,abnorm:17,abov:4,acc:17,accuraci:17,acquir:0,actual:[0,4],add:4,add_argu:17,add_modul:11,again:11,age:0,air:0,airwai:0,album:4,align:4,allergi:0,along:4,also:[0,4],altern:4,alwai:4,american:4,amet:4,amplifi:0,ani:[0,10],annot:0,anomali:17,api:[13,14,16],apnea:0,appli:[0,17],arg_typ:17,argspars:17,argument:17,argv:17,aristotl:0,arrai:17,artist:4,asbesto:0,asbestosi:0,asthma:0,att_typ:10,attach_extractor:6,audio:[0,17],augment:0,auscultatori:0,auth:0,avail:0,aveiro:0,averag:17,aviod:11,award:4,babi:4,bacon:4,balanc:[0,17],bar:4,base:[6,7,8,11,13,14,16,17],base_criterion:6,baseexplain:7,baselogg:8,baseschedul:16,basetrain:[6,11,16],basic:[5,14],basicloss:[5,14],batch:17,batch_siz:10,bcwb:17,beasti:4,beat:4,beauti:4,behavior:11,bel:4,bell:0,belli:4,below:[0,4],best:4,beta:17,between:4,billi:4,biltong:4,birthplac:4,biv:4,block:[3,4,10,12],blockquot:4,blue:4,boi:4,bold:4,bolton:4,bone:4,bool:[6,11,17],border:4,born:4,both:[0,17],box:0,bp60_heart:0,brass:4,breath:0,bresaola:4,brisket:4,brit:4,bronchial:0,bronchiol:0,bronchiti:0,brooklyn:4,brother:4,bubbl:0,build_criterion:[11,16],build_dataload:[11,16],build_dataset:[11,13,14,16],build_logg:11,build_model:[5,11,14],build_modul:[13,14],build_optim:[11,16],c_reduc:7,call:[0,11],callabl:[6,9,11],can:[0,4],cancer:0,capabl:0,captur:0,cast:4,caus:[0,11],cbam:[5,14],ccnc:17,ccwb:17,cell:4,cfg:10,challeng:[1,17],checkpoint:10,chest:0,chronic:[0,17],chuck:4,classic:4,classif:17,classifi:17,click:0,cls:11,cnc:17,cnn:10,cnn_moe:[5,14],coars:0,code:[3,4,12],coimbra:0,collect:0,color:4,common:[0,5,14],compar:0,compil:10,compile_trt:[5,14],complement:0,comput:[0,4,17],condens:4,config:[8,10,16],config_pars:[13,14],configur:10,confus:17,consid:17,consist:0,contain:[0,4],content:[4,12,13],contrast:[5,14,16],contrast_criterion:6,contrastiveloss:[5,14],contrasttrain:[6,16],conv1:10,conv2:10,conv3:10,conv3x3:10,conv4:10,conv5:10,conv6:10,convolut:10,copd:0,cord:0,correctli:17,correspond:17,countri:0,cover:0,cow:4,crackl:[0,17],crafti:4,crazi:4,creat:[4,10],crep:0,crepit:0,criterion:[9,11],criterion_opt:9,cry:4,current:[0,11],custom:[5,14],custom_model:10,customloss:[5,14],cycl:[0,17],cystic:0,darken:2,darl:4,data:[6,11,17],data_config:[11,16],data_path:0,databas:0,dataload:[6,11],dataman:[13,14],dataset:[1,11,12,16],dcnn:10,defaultdict:[6,8,11,17],definit:[4,17],demo:[5,14],demo_help:11,demonstr:4,depth:10,desir:0,detect:17,devic:[6,7,11],device_id:10,devo:4,diaphragm:0,dict:[10,17],dictionari:17,die:4,differ:0,diseas:[0,17],distinguish:4,do_explan:11,do_infer:11,document:4,dolor:4,donatello:4,doner:4,dove:4,down:4,drumstick:4,duo:4,each:[0,17],eastern:0,effect:4,either:0,electron:0,element:[3,12],els:2,emphas:0,emphysema:0,encod:0,engin:4,enough:[0,4],epic:4,epiglott:0,epoch:[6,11,16],equat:17,essua:0,eval:[5,7,14],evalu:17,everyth:17,examin:0,exampl:0,exp_id:16,expert:10,explain:[5,14],explos:0,extend:0,faculti:0,failur:0,fals:[6,7,10,11,16,17],favorit:4,fbeta:17,femal:0,fghandler:7,fibrosi:0,fig:7,fight:4,figure_to_arrai:7,file:[0,17],filet:4,filter:0,filtrat:0,find:11,fine:0,first:[0,4],fit:4,flap:0,flow:0,fluid:0,follow:4,foo:4,form:17,forward:7,fraiwan:1,framework:0,frequenc:0,from:[0,2,10],frontend:17,furthermor:0,gastroesophag:0,gener:0,generate_expid:8,gerd:0,get:4,get_all_featur:7,get_all_grad:7,get_arg:17,get_config:17,get_featur:7,get_grad:7,get_input_img:7,get_last_conv_nam:7,get_mixup_stat:17,get_precision_recall_fbeta:17,get_stat:17,girl:4,godzilla:4,gotta:4,grad_thr:[6,11],gradcamexplain:7,grai:4,grammi:4,greec:0,green:4,group:4,ham:4,happen:0,have:[0,4],head:4,header:[3,12],health:0,healthi:[0,17],heart:0,heartsound:1,henc:0,here:[0,4],hierarchi:0,high:0,higher:0,highlight:[2,4],hit:4,hnc:17,hock:4,hold:4,hookman:[5,14],hop_length:[7,11],horizont:4,hospit:0,hour:0,hover:4,how:0,html:4,human:4,hw_reduc:7,icbhi:[1,13,14,17],icbhi_basic_train:16,icbhi_contrast_train:16,icbhi_metr:[11,17],identifi:17,ill:4,imag:4,imathia:0,impos:0,in_channel:10,in_plan:10,includ:0,independ:0,index:12,indic:17,infant:0,infect:0,inform:0,inlin:[2,4],input:7,input_path:[7,11],input_shap:10,insid:[2,4],ipsum:4,issu:0,ital:4,jacki:4,jackson:4,japan:4,javascript:4,jean:4,jermain:4,join:17,jordan:0,just:0,keep:0,kid:4,kielbasa:4,king:0,kwarg:[10,11,16],kwopt:17,lab3r:0,label:[4,17],laboratori:0,ladi:4,lam:17,languag:4,larg:4,last:0,layer:7,layer_nam:7,least:0,leberka:4,left:[0,4],leonardo:4,let:4,level:2,licens:4,life:4,lighter:4,like:[0,4],limit:0,line:4,link:[1,2,4],list:[4,6,10,11],load_config:17,load_input:[7,11],loader:[13,14,17],log:8,log_path:8,logger:[5,14],logit:17,logsumexp_2d:10,loss:[5,14],low:[0,4],lower:0,lr_sched_arg:16,lung:0,macro:17,main:16,make:0,male:0,mani:0,margin:4,markdown:[3,12],marlon:4,master_cfg:11,matrix:17,max:[7,17],maximum:0,mean:11,media:4,mercuri:4,method:11,metric:[11,17],michael:4,michelangelo:4,middl:0,might:11,mignon:4,min:[7,17],min_max_scal:[7,17],mine:4,mini_batch:17,minimum:0,misc:[13,14],mix:4,mixtur:10,mixup:[11,16,17],mixup_criterion:9,mock:[6,10,11,16,17],mode:[0,7,11],model:[5,6,7,11,14,16,17],model_cfg:11,modul:[12,13],moe:10,monkei:4,more:0,most:0,move_devic:[6,11],music:4,n_fft:11,n_mel:11,n_stft:11,name:[0,4,6,7,10,11,16,17],naousa:0,narrow:0,natur:4,ncc:17,nccnc:17,nch:17,ncnc:17,nct:17,ndarrai:[6,11],nest:4,net:7,network_typ:10,nikki:4,noawardsbutthistablecelliswid:4,node:17,nois:0,non:4,nonchron:17,none:[6,7,10,11,16,17],normal:[0,4,11,17],normal_class_label:17,now:4,num_class:[10,17],num_expert:10,number:[10,17],numpi:[6,11,17],object:[0,7,8,11,16,17],obstruct:0,offici:0,old:0,one:0,oper:17,opt:17,optim:[6,11,16],option:[6,11,17],order:4,other:0,our:0,out:0,out_plan:10,output_nam:10,over:[0,4],packag:[12,13],pad:10,page:[4,12],papanikola:0,paragraph:4,parallel:[11,16],paramet:[10,11,17],parse_config:16,parse_config_dict:17,parse_config_obj:17,pascal:1,paul:4,pedro:0,peopl:0,pepa:4,per:0,perform:4,pericard:0,person:0,physician:0,pig:4,pipelin:[13,14],pitch:0,plai:4,pneumonia:0,pop:4,porchetta:4,pork:4,portug:0,poss:4,posterior:0,precis:17,prepare_kfold:11,preproc_modul:11,preprocess:[5,6,14],preprocessor:11,pretrain:10,pretti:4,princ:4,print_stat:17,problem:0,profession:0,profit:4,provid:0,publicli:0,pulmonari:0,purpl:4,rain:4,random:17,rang:0,raphael:4,rattl:0,realli:4,recal:17,record:[0,4],reflux:0,rehabilit:0,remove_handl:7,reproduc:17,requir:0,research:0,reset_al:7,reset_metr:17,reset_train:11,residualnet:10,resnet:[5,14],respiratori:[0,17],rest:0,resum:11,return_arrai:7,return_raw:11,return_stat:[6,11],rever:4,rhymin:4,rib:4,ride:4,right:4,rock:4,root:2,row:4,rsv:0,rubi:4,rule:4,rump:4,sac:0,sadaco:0,sadaco_web:12,salt:4,same:4,sampl:[0,4,7,17],sample_path:7,sample_r:11,sampler:[13,14],sausag:4,save_path:7,scale:4,schedul:[13,14],school:0,scienc:0,score:[4,17],scroll:4,search:12,second:0,seed:17,seed_everyth:17,sell:4,sensit:17,sentenc:4,seri:0,seven:0,sever:0,shade:4,shank:4,shape:17,she:4,shorter:0,should:4,shoulder:4,shown:4,side:4,sign:0,singl:4,sit:4,size:[4,8],skill:0,sleep:[0,4],slow:4,small:4,snippet:4,someth:0,somethin:4,song:4,soul:4,sound:0,soundtrack:4,sourc:[6,7,8,9,10,11,16,17],spec:[7,10],spec_displai:7,specif:17,stage:0,star:4,startin:4,stat:[11,13,14],std:11,steal:4,stealin:4,step:16,stethoscop:0,stft2meldb:11,still:4,str:[10,17],stride:10,strikethrough:4,stuck:0,style:4,subject:0,submodul:[5,13,14],subpackag:[12,13],suffer:0,support:[0,17],swell:0,symptom:0,syncyti:0,syntax:4,tabl:4,take:[4,17],task1:17,task1_1:17,task1_2:17,task2:17,task2_1:17,task2_2:17,task:[4,17],team:0,technolog:0,tensor:[10,11,17],term:4,test:11,test_basic_epoch:11,test_load:11,text:4,thei:[0,4],them:0,theme:2,thessaloniki:0,thi:[0,4,11],thing:4,three:0,thriller:4,through:[0,2,17],till:4,time:[0,4],tito:4,tmnt:4,toc:2,todo:11,too:[0,4],top:0,torch:[6,10,11,16,17],total:[0,17],train:[0,5,7,14,17],train_basic_epoch:11,train_byol:[5,14],train_config:[6,11,16],train_epoch:[11,16],train_kfold:11,train_load:[6,11],train_mixcon_epoch:6,train_stat:16,train_supcon:[5,14],trainabl:[11,16],trainer_bas:[5,6,14,16],trainer_contrast:[5,14,16],traintest:[5,6,14,16],tupl:[10,17],two:0,type:[0,11,17],under:0,underlin:4,underp:4,unexpect:11,unhealthi:17,union:[6,10,11,17],unit:0,univers:0,unless:4,unord:4,unzip:0,update_interv:[6,11],update_list:17,update_mixup_stat:17,use:4,use_wandb:11,used:[0,4],usual:0,util:[6,11,13,14],vacat:4,valid:11,valid_stat:16,validate_epoch:[11,16],varieti:0,verbos:[6,11],vertic:4,viru:0,visual:[4,5,14],vocal:[0,4],voic:0,volunt:0,wai:11,wanna:4,warner:4,wav:0,wcwb:17,web_util:[13,14],weight:6,were:0,what:4,wheez:[0,17],when:[0,4],whether:17,which:[0,11],whistl:0,white:2,whitespac:4,whom:0,wide:4,windpip:0,without:[0,17],word:0,would:4,wrap:4,wrap_model:6,y_pred:17,y_true:17,y_true_a:17,y_true_b:17,yaml:17,year:[0,4],yml_path:[10,17],you:[0,4],young:4,your:[0,4],zone:0},titles:["Datasets","SADACO","Code Blocks","SADACO_WEB","Markdown Elements","sadaco.apis package","sadaco.apis.contrastive package","sadaco.apis.explain package","sadaco.apis.logger package","sadaco.apis.losses package","sadaco.apis.models package","sadaco.apis.traintest package","Welcome to SADACO\u2019s documentation!","sadaco","sadaco package","sadaco.dataman package","sadaco.pipelines package","sadaco.utils package"],titleterms:{"2017":0,"class":0,api:[5,6,7,8,9,10,11],base:15,basic:8,basicloss:9,block:2,build_dataset:15,build_model:10,build_modul:16,cbam:10,challeng:0,cnn_moe:10,code:2,common:11,compile_trt:10,config_pars:17,content:[5,6,7,8,9,10,11,14,15,16,17],contrast:6,contrastiveloss:9,count:0,custom:10,customloss:9,data:0,dataman:15,dataset:0,demo:11,document:12,download:0,durat:0,element:4,eval:11,explain:7,fraiwan:0,header:4,heartsound:0,hookman:7,icbhi:[0,16],indic:12,link:0,loader:15,logger:8,loss:9,markdown:4,misc:17,model:10,modul:[5,6,7,8,9,10,11,14,15,16,17],packag:[5,6,7,8,9,10,11,14,15,16,17],pascal:0,pipelin:16,prepar:0,preprocess:11,resnet:10,sadaco:[1,5,6,7,8,9,10,11,12,13,14,15,16,17],sadaco_web:3,sampler:15,schedul:16,specif:0,stat:17,submodul:[6,7,8,9,10,11,15,16,17],subpackag:[5,14],tabl:12,train:11,train_byol:6,train_supcon:6,trainer_bas:11,trainer_contrast:6,traintest:11,util:17,visual:7,web_util:17,welcom:12}})