package models;

import java.awt.Rectangle;

/**
 *
 * @author Roger Schaer <roger.schaer at hevs.ch>
 */
public class SubFigure {
    private Point topLeft;
    private Point topRight;
    private Point bottomLeft;
    private Point bottomRight;

    /* Constructors */
    public SubFigure(){}

    public SubFigure(Point topLeft, Point topRight, Point bottomLeft, Point bottomRight) {
        this.topLeft = topLeft;
        this.topRight = topRight;
        this.bottomLeft = bottomLeft;
        this.bottomRight = bottomRight;
    }

    public SubFigure(Point origin, int width, int height){
        this.topLeft = origin;
        this.topRight = new Point(origin.getX() + width, origin.getY());
        this.bottomLeft = new Point(origin.getX(), origin.getY() + height);
        this.bottomRight = new Point(origin.getX() + width, origin.getY() + height);
    }

    /* Utility methods */
    public Point getCenter(){
        int centerX = (topLeft.getX() + topRight.getX()) / 2;
        int centerY = (topLeft.getY() + bottomLeft.getY()) / 2;

        return new Point(centerX, centerY);
    }

    public int getWidth(){
        return topRight.getX() - topLeft.getX();
    }

    public int getHeight(){
        return bottomLeft.getY() - topLeft.getY();
    }

    /* Calculate overlap area between a reference subfigure and a candidate */
    public double getOverlapWithOtherFigure(SubFigure other) {

        Rectangle groundTruth = new Rectangle(topLeft.getX(), topLeft.getY(), getWidth(), getHeight());
        Rectangle candidate = new Rectangle(other.topLeft.getX(), other.topLeft.getY(), other.getWidth(), other.getHeight());

        int gtArea = (int) (groundTruth.getWidth() * groundTruth.getHeight());
        int cdArea = (int) (candidate.getWidth() * candidate.getHeight());

        //Get intersection
        Rectangle intersection;
        intersection = candidate.intersection(groundTruth);

        //No intersection, no match
        if (intersection.isEmpty()) {
            return -1;
        }

        //Calculate intersection area
        int inArea = (int) (intersection.getWidth() * intersection.getHeight());

        //Return result
        return ((double) inArea / cdArea);

    }

    /* Check if there is enough overlap */
    public boolean validateAgainstOtherSubFigure(SubFigure other){
        double overlap = getOverlapWithOtherFigure(other);

        return (overlap >= (1 - Figure.MARGIN));
    }

    public void printSubFigure(){
        String symbol = "*";
        for(int i = 0; i <= (getHeight() / 100); i++){
            for(int j = 0; j < (getWidth() / 100); j++){
                System.out.print(symbol);
            }
            System.out.println();
        }
    }

    /* Standard getters & setters */
    public Point getTopLeft() {
        return topLeft;
    }

    public void setTopLeft(Point topLeft) {
        this.topLeft = topLeft;
    }

    public Point getTopRight() {
        return topRight;
    }

    public void setTopRight(Point topRight) {
        this.topRight = topRight;
    }

    public Point getBottomLeft() {
        return bottomLeft;
    }

    public void setBottomLeft(Point bottomLeft) {
        this.bottomLeft = bottomLeft;
    }

    public Point getBottomRight() {
        return bottomRight;
    }

    public void setBottomRight(Point bottomRight) {
        this.bottomRight = bottomRight;
    }
}
