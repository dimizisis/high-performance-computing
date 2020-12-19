
class PiMonteCarloParallelSharedArray {
  public static void main(String[] args) throws InterruptedException {
    if (args.length != 1) {
	    System.out.println("Usage : PiMonteCarloParallelSharedArray <num points>");
      System.exit(1);
    }
    int niter = Integer.parseInt(args[0]);
    int[] counts = new int[niter];
    int count = 0;

    CalcThread[] threads = new CalcThread[niter];
    
    /* Create threads */
    for (int i=0; i<niter; ++i)
      threads[i] = new CalcThread(counts, i);

    /* Start threads */
    for (int i=0; i<niter; ++i)
      threads[i].start();

    /* Wait for all threads to finish */
    for (int i=0; i<niter; ++i)
      threads[i].join();

    /* Perform reduction */
    for (int i=0; i<niter; ++i)
      count += counts[i];
    
    /* Final calculation (pi) */
    double pi = (double) count/niter*4;

    /* Print results */
    System.out.println(pi);
  }
  
  private static class CalcThread extends Thread{

    int[] counts;
    int i;

    public CalcThread(int[] counts, int i){
      this.counts = counts;
      this.i = i;
    }

    @Override
    public void run(){

      double x, y, z;
      x = Math.random();
      y = Math.random();
      z = x*x+y*y;
      if(z<=1)
        ++counts[i];
    }

  }
}