/* PART 2: SQLite
/* We now want you to jump over to a local instance of the database on your machine. 

Copy and paste the LocalSQLConnection.py script into an empty Jupyter notebook, and run it. 

Make sure that the SQLFiles folder containing thes files is in your working directory, and
that you haven't changed the name of the .db file from 'sqlite\db\pythonsqlite'.

You should see the output from the initial query 'SELECT * FROM FACILITIES'.

Complete the remaining tasks in the Jupyter interface. If you struggle, feel free to go back
to the PHPMyAdmin interface as and when you need to. 

You'll need to paste your query into value of the 'query1' variable and run the code block again to get an output.
 
QUESTIONS:
/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */

        SELECT name, total_revenue
        FROM (
            SELECT b.name,
            SUM(CASE WHEN b.memid = 0 THEN f.guestcost * b.slots ELSE f.membercost * b.slots END ) AS total_revenue
            FROM Bookings AS b
            JOIN Facilities AS f ON b.facid = f.facid
            GROUP BY f.name
            ) 
        AS facilities_revenue
        WHERE total_revenue < 1000
        ORDER BY total_revenue;

/* Q11: Produce a report of members and who recommended them in alphabetic surname,firstname order */

		SELECT m.memid, m.recommendedby, 
            m.surname || ", " || m.firstname AS member_name
        FROM Members as m
        LEFT OUTER JOIN Members as n
        ON m.recommendedby = n.memid
        WHERE m.surname != 'GUEST'
        ORDER BY m.recommendedby, member_name

/* Q12: Find the facilities with their usage by member, but not guests */

		SELECT f.facid, f.name, m.memid,
            m.surname || ", " || m.firstname AS member_name,
            b.starttime
        FROM Members AS m
        JOIN Bookings AS b ON m.memid = b.memid
        JOIN Facilities AS f ON b.facid = f.facid
        WHERE m.memid != 0
        GROUP BY f.facid
        ORDER BY member_name;

/* Q13: Find the facilities usage by month, but not guests */;

		SELECT f.facid, f.name, m.memid,
            m.surname || ", " || m.firstname AS member_name,
            b.starttime
        FROM Members AS m
        JOIN Bookings AS b ON m.memid = b.memid
        JOIN Facilities AS f ON b.facid = f.facid
        WHERE m.memid != 0
        GROUP BY f.facid
        ORDER BY b.starttime;