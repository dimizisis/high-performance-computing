
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

class PiMonteCarloParallelLock {
    public static void main(String[] args) throws InterruptedException {
      if (args.length != 1) {
          System.out.println("Usage : PiMonteCarloParallelLock <num points>");
        System.exit(1);
      }
      int niter = Integer.parseInt(args[0]);
      int[] count = {0}; // one-element array

      Lock lock = new ReentrantLock();
  
      CalcThread[] threads = new CalcThread[niter];
      
      /* Create threads */
      for (int i=0; i<niter; ++i)
        threads[i] = new CalcThread(lock, count);
  
      /* Start threads */
      for (int i=0; i<niter; ++i)
        threads[i].start();
  
      /* Wait for all threads to finish */
      for (int i=0; i<niter; ++i)
        threads[i].join();
  
      /* Final calculation (pi) */
      double pi = (double) count[0]/niter*4;

      /* Print results */
      System.out.println(pi);
    }
    
    private static class CalcThread extends Thread{
  
        private Lock lock;
        private int[] count;
  
        public CalcThread(Lock lock, int[] count){
            this.lock = lock;
            this.count = count;
        }
  
        @Override
        public void run(){
  
            double x, y, z;
            x = Math.random();
            y = Math.random();
            z = x*x+y*y;
            if(z<=1){
                lock.lock();
                ++count[0];
                lock.unlock();
            }
        }
  
    }
}