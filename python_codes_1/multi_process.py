from multiprocessing import Process

def factorial(num):
    n=1
    while num>1:
        n*=num
        num=num-1
    #return n
    print(n)


def Main():
    jobs=[]
    #run the same fucntion with different subsets of the image and then join the result
    p1=Process(target=factorial, args=(100,))
    p2=Process(target=factorial, args=(100,))
    #p3=Process(target=, args=())
    #p4=Process(target=, args=())
    jobs=[p1, p2]
    p1.start()
    p2.start()
    result=[]
    for proc in jobs:
        proc.join()
        result.append(proc.exitcode)
    #p1.join()
    #p2.join()
    print(jobs)
    
if __name__=='__main__':
    print(Main())