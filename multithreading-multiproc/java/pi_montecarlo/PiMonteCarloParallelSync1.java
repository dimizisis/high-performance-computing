
class PiMonteCarloParallelSync1 {
    public static void main(String[] args) throws InterruptedException {
      if (args.length != 1) {
          System.out.println("Usage : PiMonteCarloParallelSync1 <num points>");
        System.exit(1);
      }
      int niter = Integer.parseInt(args[0]);
      int[] count = {0}; // one-element array
  
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
      double pi = (double) count[0]/niter*4;

      /* Print results */
      System.out.println(pi);
    }
    
    private static class CalcThread extends Thread{
  
        private int[] count;
  
        public CalcThread(int[] count){
            this.count = count;
        }
  
        @Override
        public void run(){
            double x, y, z;
            x = Math.random();
            y = Math.random();
            z = x*x+y*y;
            if(z<=1){
                synchronized(this.getClass()){
                    ++count[0];
                }
            }
        }
  
    }
}