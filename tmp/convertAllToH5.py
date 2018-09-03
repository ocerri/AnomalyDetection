import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()    
    parser.add_argument('-i','--input', type=str, help="input  EOS directory", required=True)
    parser.add_argument('-q', '--queue', type=str, default = "1nw", help="LSFBATCH queue name")

    args = parser.parse_args()

    # create input list
    listName = args.input.split("/")[-1]
    os.system("ls %s | grep root > %s.list" %(args.input,listName))
    # list of existing files  
    os.system("ls %s | grep h5 > %s_existing.list" %(args.input, listName))
    f = open("%s_existing.list" %listName)
    existing = f.readlines()
    f.close()
    #
    listIN = open("%s.list" %listName)
    i = 0 
    os.system("mkdir %s" %listName)
    if not os.path.isdir(listName): os.mkdir(listName)
    for fileName in listIN:
        myjob = fileName[:-1]
        if myjob.replace(".root",".h5\n") in existing: 
            i = i + 1
            continue
        mydir = args.input+"/"
        script = open("%s/%s" %(listName,myjob.replace(".root","_h5.src")), 'w')
        script.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_89/x86_64-slc6-gcc62-opt/setup.sh\n")  
        script.write("python %s/script/convertToH5.py %s/%s\n" %(os.getcwd(),mydir,fileName[:-1]))
        script.close()
        os.system("bsub -q %s -o %s/%s_%i.log -J %s_%i < %s/%s" %(args.queue, listName, listName, i, listName, i, listName, myjob.replace(".root","_h5.src")));
        print "submitting job n. %i to the queue %s...\n" %(i,args.queue)
        i = i+1
    listIN.close()
