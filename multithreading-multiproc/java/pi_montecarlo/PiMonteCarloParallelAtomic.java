import java.util.concurrent.atomic.AtomicInteger;

class PiMonteCarloParallelAtomic {
    public static void main(String[] args) throws InterruptedException {
      if (args.length != 1) {
          System.out.println("Usage : PiMonteCarloParallelAtomic <num points>");
        System.exit(1);
      }
      int niter = Integer.parseInt(args[0]);
      AtomicInteger count = new AtomicInteger(0);
  
      CalcThread[] threads = new CalcThread[niter];
      
      /* Create threads */
      for (int i=0; i<niter; ++i)
        threads[i] = new CalcThread(count);
  
      /* Start threads */
      for (int i=0; i<niter; ++i)
        threads[i].start();
  
      /* Wait for all threads to finish */
      for (int i=0; i<niter; ++i)
        threads[i].join();
  
      /* Final calculation (pi) */
      double pi = (double) count.get()/niter*4;

      /* Print results */
      System.out.println(pi);
    }
    
    private static class CalcThread extends Thread{
  
        AtomicInteger count;
  
        public CalcThread(AtomicInteger count){
            this.count = count;
        }
  
        @Override
        public void run(){
  
            double x, y, z;
            x = Math.random();
            y = Math.random();
            z = x*x+y*y;
            if(z<=1)
                count.incrementAndGet();
        }
  
    }
}