package models;

import java.util.Comparator;

/**
 *
 * @author Roger Schaer <roger.schaer at hevs.ch>
 */
public class Point {

    public static final Comparator<Point> X_COMPARATOR = new XComparator();
    public static final Comparator<Point> Y_COMPARATOR = new YComparator();

    private int x;
    private int y;

    public Point(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public int getX() {
        return x;
    }

    public void setX(int x) {
        this.x = x;
    }

    public int getY() {
        return y;
    }

    public void setY(int y) {
        this.y = y;
    }

    private static class XComparator implements Comparator<Point>{
        @Override
        public int compare(Point o1, Point o2) {
            return Integer.compare(o1.x, o2.x);
        }
    }

    private static class YComparator implements Comparator<Point>{
        @Override
        public int compare(Point o1, Point o2) {
            return Integer.compare(o1.y, o2.y);
        }
    }

    public String toString(){
        return "X : " + x + ", Y : " + y;
    }


}
